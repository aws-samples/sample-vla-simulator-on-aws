/**
 * AzSelectorConstruct
 *
 * Custom Resource Lambda를 사용하여 probe 후 즉시 실 인스턴스를 launch.
 * probe + terminate → launch 사이의 레이스 컨디션 제거.
 *
 * 동작 방식:
 * 1. 인스턴스 타입 fallback 리스트를 순차 시도
 * 2. 각 인스턴스 타입에 대해 describe-instance-type-offerings로 지원 AZ 목록 조회
 * 3. AZ 목록을 셔플하여 특정 AZ 집중 방지
 * 4. 각 AZ에서 RunInstances (UserData + IAM + SG 포함) 직접 실행
 * 5. 성공하면 InstanceId + AZ + InstanceType 반환 (probe 불필요)
 * 6. InsufficientInstanceCapacity이면 다음 AZ 시도
 * 7. 해당 타입의 모든 AZ 실패 시 다음 인스턴스 타입으로 fallback
 * 8. Delete: InstanceId로 terminate
 */
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';
import { NagSuppressions } from 'cdk-nag';

export interface AzSelectorProps {
  instanceTypes: string[];
  amiId: string;
  subnetIds: string[];
  // Real instance launch params (replaces probe+terminate)
  userDataBase64: string;
  iamInstanceProfileArn: string;
  securityGroupIds: string[];
  ebsGb: number;
  tags: Array<{ key: string; value: string }>;
}

export class AzSelectorConstruct extends Construct {
  public readonly instanceId: string;
  public readonly availabilityZone: string;
  public readonly resolvedInstanceType: string;
  public readonly subnetId: string;

  constructor(scope: Construct, id: string, props: AzSelectorProps) {
    super(scope, id);

    const lambdaRole = new iam.Role(this, 'AzSelectorRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
      inlinePolicies: {
        AzSelectorPolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'ec2:DescribeInstanceTypeOfferings',
                'ec2:DescribeSubnets',
                'ec2:RunInstances',
                'ec2:TerminateInstances',
                'ec2:DescribeInstances',
                'ec2:CreateTags',
              ],
              resources: ['*'],
            }),
            new iam.PolicyStatement({
              actions: ['iam:PassRole'],
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    const azSelectorFn = new lambda.Function(this, 'AzSelectorFunction', {
      runtime: lambda.Runtime.PYTHON_3_13,
      handler: 'index.handler',
      role: lambdaRole,
      timeout: cdk.Duration.minutes(10),
      code: lambda.Code.fromInline(AZ_SELECTOR_LAMBDA_CODE),
      description: 'Launches GPU instance with capacity fallback across AZs/types',
    });

    const customResource = new cdk.CustomResource(this, 'AzSelectorResource', {
      serviceToken: azSelectorFn.functionArn,
      properties: {
        InstanceTypes: props.instanceTypes.join(','),
        AmiId: props.amiId,
        SubnetIds: props.subnetIds.join(','),
        UserDataBase64: props.userDataBase64,
        IamInstanceProfileArn: props.iamInstanceProfileArn,
        SecurityGroupIds: props.securityGroupIds.join(','),
        EbsGb: props.ebsGb.toString(),
        Tags: JSON.stringify(props.tags),
        Timestamp: Date.now().toString(),
      },
    });

    azSelectorFn.addPermission('CfnInvoke', {
      principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
    });

    this.instanceId = customResource.getAttString('InstanceId');
    this.availabilityZone = customResource.getAttString('AvailabilityZone');
    this.resolvedInstanceType = customResource.getAttString('InstanceType');
    this.subnetId = customResource.getAttString('SubnetId');

    NagSuppressions.addResourceSuppressions(lambdaRole, [
      {
        id: 'AwsSolutions-IAM4',
        reason: 'AWSLambdaBasicExecutionRole is the minimal managed policy for Lambda CloudWatch Logs access.',
        appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'AzSelector Lambda requires broad EC2 permissions to launch instances across AZs at deploy time. Resource scoping is not feasible as instance IDs and AZs are unknown at synth time.',
        appliesTo: ['Resource::*'],
      },
    ]);

    NagSuppressions.addResourceSuppressions(azSelectorFn, [
      {
        id: 'AwsSolutions-L1',
        reason: 'Python 3.13 is the latest stable production runtime for AWS Lambda.',
      },
    ]);
  }
}

const AZ_SELECTOR_LAMBDA_CODE = `
import json
import boto3
import random
import cfnresponse

def handler(event, context):
    print(json.dumps({k: v for k, v in event.items() if k != 'ResourceProperties'} | {'ResourceProperties': {k: (v[:80]+'...' if isinstance(v,str) and len(v)>80 else v) for k,v in event.get('ResourceProperties',{}).items()}}))

    region = context.invoked_function_arn.split(':')[3]
    ec2 = boto3.client('ec2', region_name=region)

    if event['RequestType'] == 'Delete':
        instance_id = event.get('PhysicalResourceId', '')
        if instance_id and instance_id.startswith('i-'):
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
                print(f'Terminated instance {instance_id}')
            except Exception as e:
                print(f'Terminate failed (ignoring): {e}')
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {},
            physicalResourceId=instance_id or 'az-selector-deleted')
        return

    try:
        props = event['ResourceProperties']
        instance_types = props['InstanceTypes'].split(',')
        ami_id = props['AmiId']
        subnet_ids = props['SubnetIds'].split(',')
        user_data_b64 = props['UserDataBase64']
        iam_profile_arn = props['IamInstanceProfileArn']
        sg_ids = props['SecurityGroupIds'].split(',')
        ebs_gb = int(props['EbsGb'])
        tags = json.loads(props['Tags'])

        subnets_resp = ec2.describe_subnets(SubnetIds=subnet_ids)
        subnet_az_map = {s['SubnetId']: s['AvailabilityZone'] for s in subnets_resp['Subnets']}
        print(f'Subnet AZ map: {subnet_az_map}')

        all_tried = []

        for instance_type in instance_types:
            print(f'--- Trying instance type: {instance_type} ---')

            resp = ec2.describe_instance_type_offerings(
                LocationType='availability-zone',
                Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
            )
            supported_azs = {o['Location'] for o in resp['InstanceTypeOfferings']}
            print(f'Supported AZs for {instance_type}: {supported_azs}')

            candidates = [(sid, az) for sid, az in subnet_az_map.items() if az in supported_azs]
            if not candidates:
                print(f'{instance_type} not available in any VPC subnet AZ, skipping...')
                all_tried.append(f'{instance_type}(no subnet AZ match)')
                continue

            random.shuffle(candidates)

            for subnet_id, az in candidates:
                print(f'Launching {instance_type} in {az} (subnet: {subnet_id})')
                try:
                    tag_specs = [{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': t['key'], 'Value': t['value']} for t in tags]
                    }]
                    run_resp = ec2.run_instances(
                        InstanceType=instance_type,
                        ImageId=ami_id,
                        MinCount=1,
                        MaxCount=1,
                        SubnetId=subnet_id,
                        SecurityGroupIds=sg_ids,
                        IamInstanceProfile={'Arn': iam_profile_arn},
                        UserData=user_data_b64,
                        BlockDeviceMappings=[{
                            'DeviceName': '/dev/sda1',
                            'Ebs': {
                                'VolumeSize': ebs_gb,
                                'VolumeType': 'gp3',
                                'DeleteOnTermination': True,
                                'Encrypted': True,
                            }
                        }],
                        TagSpecifications=tag_specs,
                    )
                    instance_id = run_resp['Instances'][0]['InstanceId']
                    print(f'SUCCESS: {instance_type} in {az} (instance: {instance_id})')

                    cfnresponse.send(event, context, cfnresponse.SUCCESS,
                        {'InstanceId': instance_id, 'AvailabilityZone': az,
                         'InstanceType': instance_type, 'SubnetId': subnet_id},
                        physicalResourceId=instance_id)
                    return

                except Exception as e:
                    error_msg = str(e)
                    if 'InsufficientInstanceCapacity' in error_msg:
                        print(f'InsufficientCapacity: {instance_type} in {az}')
                        all_tried.append(f'{instance_type}/{az}')
                        continue
                    elif 'Unsupported' in error_msg:
                        print(f'Unsupported: {instance_type} in {az}')
                        all_tried.append(f'{instance_type}/{az}(unsupported)')
                        continue
                    else:
                        print(f'Unexpected error: {e}')
                        raise

            print(f'All AZs exhausted for {instance_type}, falling back...')

        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=f'No capacity available for any instance type: {all_tried}',
            physicalResourceId='az-selector-failed')

    except Exception as e:
        print(f'Error: {e}')
        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=str(e),
            physicalResourceId='az-selector-error')
`;
