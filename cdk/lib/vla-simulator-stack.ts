/**
 * vla-simulator Unified CDK Stack
 *
 * Supports GR00T and π0.5 via a single stack parameterised by `vla` context.
 *
 * Resources: VPC + SG + S3 + SNS + IAM + AzSelector + EC2
 * Deploy:    python deploy.py --vla gr00t [--bridge]
 *            python deploy.py --vla pi    [--bridge]
 * Teardown:  python destroy.py --vla {gr00t|pi}
 *
 * Config entry point: ../simulator-config.yaml + ../models/{vla}.yaml
 */

import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import { NagSuppressions } from 'cdk-nag';

import { AzSelectorConstruct } from './az-selector';

export interface VlaSimulatorStackProps extends cdk.StackProps {
  vla: string;
  /** OpenVLA-OFT only: selected LIBERO suite (spatial | object | goal | 10). */
  liberoSuite?: string;
}

/**
 * DLAMI AMI mapping (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1)
 * Pre-installed: NVIDIA driver 550 + CUDA + Docker + Python
 * ap-southeast-1 (Singapore) excluded — only T4/V100 (pre-Ampere, FlashAttention incompatible).
 *
 * To add a new region: EC2 console → AMIs → Public images →
 *   search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * Ubuntu 22.04"
 */
const DLAMI_MAPPING: Record<string, string> = {
  'ap-northeast-2': 'ami-0e8f4fdb799f677ca', // Seoul     (PyTorch 2.4.1, 2025-06-24)
  'ap-northeast-1': 'ami-01e1a167769a23572', // Tokyo     (PyTorch 2.4.1)
  'us-east-1':      'ami-0aee7b90d684e107d', // Virginia  (PyTorch 2.4.1)
  'us-west-2':      'ami-0ed3bd866951103a1', // Oregon    (PyTorch 2.4.1)
  'eu-central-1':   'ami-03aa80bc63bbd3638', // Frankfurt (PyTorch 2.4.1)
};

const INSTANCE_TYPES: Record<string, string[]> = {
  gr00t:         ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  'gr00t-gr1':   ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  // G1 WBC rollout = 5 parallel MuJoCo envs (CPU/EGL render) + 1 CUDA policy server.
  // 12xlarge tier mirrors gr00t-gr1; xlarge single-GPU kept last as a capacity fallback.
  'gr00t-g1':    ['g6.12xlarge', 'g5.12xlarge', 'g6.xlarge',  'g5.xlarge'],
  pi:            ['g5.xlarge',   'g5.2xlarge',  'g6.xlarge',  'g6.2xlarge'],
  'openvla-oft': ['g6.xlarge',   'g5.xlarge',   'g6.2xlarge', 'g5.2xlarge'],
  lap:           ['g6.xlarge',   'g6.2xlarge',  'g5.xlarge',  'g5.2xlarge'],
  // Phase 0 GATE PASS (2026-06-19): eager rollout peak VRAM ~17GB on L40S(sm_89) → L4 24GB FITS
  // (~6.6GB headroom). Single-GPU xlarge sufficient; .2xlarge offers more system RAM for n_envs>1.
  rldx:          ['g6.xlarge',   'g6.2xlarge',  'g5.xlarge',  'g5.2xlarge'],
  // Isaac Sim 5.1 + --enable_cameras render is heavy → 12xlarge tier (mirrors gr00t-gr1).
  'openarm-isaac': ['g6.12xlarge', 'g5.12xlarge', 'g6.2xlarge', 'g5.2xlarge'],
  // 16 collection envs × 2 TiledCameras → ResourceLoader needs >24 GB device mem, so a single-GPU
  // L4/A10G (24 GB) OOMs at first render (run 0418: ERROR_OUT_OF_DEVICE_MEMORY before SM loop).
  // MULTI-GPU ONLY (≥4 GPU): never let AzSelector fall back to a single-GPU type. 24xlarge is the
  // capacity fallback (also 4 GPU, more vCPU). g6/g5 .16xlarge is a single-GPU trap — excluded.
  // CAPACITY (2026-06-11, run9): all 16 g5/g6 .12xl/.24xl × AZ combos returned InsufficientInstance-
  // Capacity in us-east-1 → 0-cost rollback (AzSelector probes via run_instances BEFORE launch).
  // Widened with g6e (4× L40S 48 GB — 4-GPU, MORE VRAM than L4, kept LAST so cheaper g5/g6 win first).
  'openarm-lift-act': ['g6.12xlarge', 'g5.12xlarge', 'g6.24xlarge', 'g5.24xlarge', 'g6e.12xlarge', 'g6e.24xlarge'],
};

export class VlaSimulatorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: VlaSimulatorStackProps) {
    super(scope, id, props);

    const { vla, liberoSuite } = props;

    // ── Load config ──────────────────────────────────────────────
    const simConfigPath = path.resolve(__dirname, '../../simulator-config.yaml');
    const modelConfigPath = path.resolve(__dirname, `../../models/${vla}.yaml`);
    const simConfig = yaml.load(fs.readFileSync(simConfigPath, 'utf8')) as any;
    const modelConfig = yaml.load(fs.readFileSync(modelConfigPath, 'utf8')) as any;
    const deployment = { ...simConfig.deployment, ...(modelConfig.deployment ?? {}) };
    const instanceCfg = modelConfig.instance ?? {};

    // CDK context takes precedence (injected by deploy.py)
    const notifyEmail: string = this.node.tryGetContext('notify_email') ?? deployment.notify_email;
    const s3ResultsPrefix: string = deployment.s3_results_prefix ?? 'vla-sim-results';
    const autoTerminate: boolean = deployment.auto_terminate ?? true;
    const stackName = id;

    // Bridge mode: vpc_id context set → import existing VPC
    const vpcIdCtx: string | undefined =
      (this.node.tryGetContext('vpc_id') as string | undefined) || undefined;

    // π0.5 bridge additionally needs nlb_endpoint
    const nlbEndpoint: string =
      (this.node.tryGetContext('nlb_endpoint') as string | undefined) ?? '';
    const isGr00tVla = vla === 'gr00t' || vla === 'gr00t-gr1' || vla === 'gr00t-g1';
    const bridgeMode = !!(vpcIdCtx && (isGr00tVla || nlbEndpoint));

    // EBS size: bridge mode may use smaller disk (π0.5 skips checkpoint download)
    const ebsGb: number = bridgeMode
      ? (instanceCfg.ebs_gb_bridge ?? instanceCfg.ebs_gb ?? 200)
      : (instanceCfg.ebs_gb ?? 200);

    // CreationPolicy timeout from model config. For openvla-oft this is a suite-keyed
    // map (PT120M for short-horizon, PT240M for LIBERO-10); for others it's a scalar.
    const rawTimeout: string | Record<string, string> = instanceCfg.creationpolicy_timeout ?? 'PT180M';
    let creationTimeout: string;
    if (typeof rawTimeout === 'string') {
      creationTimeout = rawTimeout;
    } else {
      const suiteKey = liberoSuite || '10';
      creationTimeout = rawTimeout[suiteKey] ?? rawTimeout['10'] ?? 'PT180M';
    }

    // ── AMI ───────────────────────────────────────────────────────
    const region = this.region;
    const amiId = DLAMI_MAPPING[region];
    if (!amiId) {
      throw new Error(
        `Unsupported region: ${region}\n` +
        `Supported regions: ${Object.keys(DLAMI_MAPPING).join(', ')}\n` +
        `To add a new region, update DLAMI_MAPPING in cdk/lib/vla-simulator-stack.ts.`,
      );
    }

    // ── VPC ───────────────────────────────────────────────────────
    const vpcLogicalId = `${vla.toUpperCase()}VPC`;
    let vpc: ec2.IVpc;
    if (vpcIdCtx) {
      vpc = ec2.Vpc.fromLookup(this, vpcLogicalId, { vpcId: vpcIdCtx });
    } else {
      vpc = new ec2.Vpc(this, vpcLogicalId, {
        maxAzs: 4,
        natGateways: 1,
        subnetConfiguration: [
          { name: 'public',  subnetType: ec2.SubnetType.PUBLIC,               cidrMask: 24 },
          { name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,  cidrMask: 24 },
        ],
      });
    }

    // ── Security Group ────────────────────────────────────────────
    const sg = new ec2.SecurityGroup(this, `${vla.toUpperCase()}SG`, {
      vpc,
      description: `${vla} simulator instance - outbound only`,
      allowAllOutbound: true,
    });

    // ── S3 bucket ─────────────────────────────────────────────────
    const bucket = new s3.Bucket(this, 'ResultsBucket', {
      bucketName: `${s3ResultsPrefix}-${stackName.toLowerCase()}-${this.region}-${this.account}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      enforceSSL: true,
    });

    // ── SNS ───────────────────────────────────────────────────────
    const topic = new sns.Topic(this, 'NotificationTopic', {
      topicName: `${stackName}-${vla}-notify`,
    });
    new sns.Subscription(this, 'EmailSubscription', {
      topic,
      protocol: sns.SubscriptionProtocol.EMAIL,
      endpoint: notifyEmail,
    });

    // ── IAM Role ──────────────────────────────────────────────────
    const logGroupPrefix = `/${vla}/*`;
    const role = new iam.Role(this, `${vla.charAt(0).toUpperCase() + vla.slice(1)}InstanceRole`, {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
      inlinePolicies: {
        VlaSimPolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: ['s3:PutObject', 's3:PutObjectAcl'],
              resources: [`${bucket.bucketArn}/*`],
            }),
            new iam.PolicyStatement({
              actions: ['sns:Publish'],
              resources: [topic.topicArn],
            }),
            new iam.PolicyStatement({
              actions: ['ec2:TerminateInstances'],
              resources: ['*'],
              conditions: { StringEquals: { 'ec2:ResourceTag/StackName': stackName } },
            }),
            new iam.PolicyStatement({
              actions: ['cloudformation:SignalResource'],
              resources: ['*'],
            }),
            new iam.PolicyStatement({
              actions: [
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
                'logs:DescribeLogStreams',
              ],
              resources: [
                `arn:${cdk.Aws.PARTITION}:logs:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:log-group:${logGroupPrefix}`,
              ],
            }),
            new iam.PolicyStatement({
              actions: ['ssm:GetParameter'],
              resources: [
                `arn:${cdk.Aws.PARTITION}:ssm:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:parameter/vla-simulator/*`,
              ],
            }),
            new iam.PolicyStatement({
              actions: ['kms:Decrypt'],
              resources: [`arn:${cdk.Aws.PARTITION}:kms:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:key/*`],
              conditions: { StringEquals: { 'kms:ViaService': `ssm.${cdk.Aws.REGION}.amazonaws.com` } },
            }),
          ],
        }),
      },
    });

    const instanceProfile = new iam.CfnInstanceProfile(this, `${vla.toUpperCase()}InstanceProfile`, {
      roles: [role.roleName],
    });

    // ── WaitCondition (replaces CfnInstance CreationPolicy) ───────
    // UserData sends SUCCESS/FAILURE to the presigned URL instead of cfn signal-resource.
    const waitHandle = new cdk.CfnWaitConditionHandle(this, `${vla.toUpperCase()}WaitHandle`);
    const waitCondition = new cdk.CfnWaitCondition(this, `${vla.toUpperCase()}WaitCondition`, {
      count: 1,
      timeout: String(
        parseInt(creationTimeout.replace(/\D/g, '')) * 60  // PT180M → 10800 seconds
      ),
      handle: waitHandle.ref,
    });

    // ── UserData (S3 Asset bootstrap) ─────────────────────────────
    const userDataPath = path.resolve(__dirname, `../../assets/userdata/${vla}.sh`);
    if (!fs.existsSync(userDataPath)) {
      throw new Error(
        `UserData script not found: ${userDataPath}\n` +
        `Use deploy.py instead of cdk deploy directly — it generates ${vla}.sh automatically.\n` +
        `Or run manually: python generate.py --vla ${vla}`,
      );
    }
    const userDataAsset = new s3assets.Asset(this, `${vla.toUpperCase()}UserDataAsset`, {
      path: userDataPath,
    });
    userDataAsset.grantRead(role);

    // Bootstrap script: inject WaitCondition presigned URL instead of CFN signal resource
    const baseBootstrapLines = [
      '#!/bin/bash',
      `# vla-simulator ${vla} — S3 bootstrap`,
      'export S3BucketName="${S3BucketName}"',
      'export SnsTopicArn="${SnsTopicArn}"',
      'export StackName="${StackName}"',
      'export WaitConditionUrl="${WaitConditionUrl}"',
      'export Region="${Region}"',
      'export AutoTerminate="${AutoTerminate}"',
    ];

    const bootstrapSubVars: Record<string, string> = {
      S3BucketName:     bucket.bucketName,
      SnsTopicArn:      topic.topicArn,
      StackName:        stackName,
      WaitConditionUrl: waitHandle.ref,
      Region:           this.region,
      AutoTerminate:    autoTerminate ? 'true' : 'false',
    };

    if (vla === 'pi' || vla === 'lap') {
      // pi & lap both run the openpi WebSocket↔gRPC bridge in bridge mode.
      baseBootstrapLines.push(
        'export BridgeMode="${BridgeMode}"',
        'export NlbEndpoint="${NlbEndpoint}"',
      );
      bootstrapSubVars['BridgeMode'] = bridgeMode ? 'true' : 'false';
      bootstrapSubVars['NlbEndpoint'] = nlbEndpoint;
    }
    // gr00t-gr1 uses the same bootstrap as gr00t (no extra variables needed)

    baseBootstrapLines.push(
      `aws s3 cp ${userDataAsset.s3ObjectUrl} /tmp/${vla}.sh`,
      `chmod +x /tmp/${vla}.sh`,
      `exec /tmp/${vla}.sh`,
    );

    const userDataRendered = cdk.Fn.base64(
      cdk.Fn.sub(baseBootstrapLines.join('\n'), bootstrapSubVars),
    );

    // ── AzSelector: launch real instance (no probe+terminate race) ─
    const azSelector = new AzSelectorConstruct(this, 'AzSelector', {
      instanceTypes: INSTANCE_TYPES[vla] ?? INSTANCE_TYPES['gr00t'],
      amiId,
      subnetIds: vpc.privateSubnets.map((s) => s.subnetId),
      userDataBase64: userDataRendered,
      iamInstanceProfileArn: instanceProfile.attrArn,
      securityGroupIds: [sg.securityGroupId],
      ebsGb,
      tags: [
        { key: 'Name',      value: `${stackName}-${vla}` },
        { key: 'StackName', value: stackName },
      ],
    });
    // WaitCondition timeout starts when CFN processes it. The instance is launched by AzSelector
    // Lambda. No explicit DependsOn needed — WaitConditionHandle URL is embedded in UserData,
    // so the instance will signal back once the script completes.

    // ── Outputs ───────────────────────────────────────────────────
    new cdk.CfnOutput(this, 'S3BucketName', {
      value: bucket.bucketName,
      description: `Results S3 bucket (download: aws s3 sync s3://BUCKET/RUN_ID/ ./${vla}-results/RUN_ID/)`,
    });
    new cdk.CfnOutput(this, 'SNSTopicArn', {
      value: topic.topicArn,
      description: 'Completion notification SNS topic ARN',
    });
    new cdk.CfnOutput(this, 'InstanceId', {
      value: azSelector.instanceId,
      description: 'EC2 instance ID',
    });
    new cdk.CfnOutput(this, 'SelectedInstanceType', {
      value: azSelector.resolvedInstanceType,
      description: 'GPU instance type selected by AzSelector',
    });
    new cdk.CfnOutput(this, 'SelectedAZ', {
      value: azSelector.availabilityZone,
      description: 'Availability zone selected by AzSelector',
    });
    new cdk.CfnOutput(this, 'NotifyEmail', {
      value: notifyEmail,
      description: 'Completion notification email address',
    });
    if (vla === 'pi' || vla === 'lap') {
      new cdk.CfnOutput(this, 'Mode', {
        value: bridgeMode ? `bridge (NLB: ${nlbEndpoint})` : 'local',
        description: 'Deployment mode: local (Docker Compose) or bridge (gRPC NLB)',
      });
    }

    // ── cdk-nag suppressions ──────────────────────────────────────
    // EC2 instance is launched by AzSelector Lambda (not a CfnInstance resource)
    // EC28/EC29 nag rules do not apply to Lambda-launched instances.

    if (!vpcIdCtx) {
      NagSuppressions.addResourceSuppressionsByPath(this, `/${id}/${vpcLogicalId}/Resource`, [
        { id: 'AwsSolutions-VPC7', reason: 'VPC Flow Logs omitted for this sample to reduce cost. Short-lived stack with no inbound traffic allowed.' },
      ]);
    }

    NagSuppressions.addResourceSuppressions(bucket, [
      { id: 'AwsSolutions-S1', reason: 'Server access logging omitted for this sample results bucket. Bucket is private (BlockPublicAccess.BLOCK_ALL).' },
    ]);

    NagSuppressions.addResourceSuppressions(topic, [
      { id: 'AwsSolutions-SNS3', reason: 'SNS topic used only for email notifications to end users. Enforcing SSL on publish not required for this notification-only use case.' },
    ]);

    const roleLogicalId = `${vla.charAt(0).toUpperCase() + vla.slice(1)}InstanceRole`;
    NagSuppressions.addResourceSuppressions(role, [
      { id: 'AwsSolutions-IAM4', reason: 'AmazonSSMManagedInstanceCore enables SSM Session Manager for browser-based terminal access.', appliesTo: ['Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore'] },
      { id: 'AwsSolutions-IAM5', reason: 'Wildcard on results bucket objects required for dynamic output file names.', appliesTo: ['Resource::<ResultsBucketA95A2103.Arn>/*'] },
      { id: 'AwsSolutions-IAM5', reason: 'ec2:TerminateInstances constrained by StringEquals on StackName tag. cloudformation:SignalResource cannot be scoped at synth time.', appliesTo: ['Resource::*'] },
      {
        id: 'AwsSolutions-IAM5',
        reason: `CloudWatch Logs wildcard limited to /${vla}/* log group prefix.`,
        appliesTo: [`Resource::arn:<AWS::Partition>:logs:<AWS::Region>:<AWS::AccountId>:log-group:/${vla}/*`],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'KMS key ARN cannot be determined at synth time; scoped to SSM service via kms:ViaService condition.',
        appliesTo: [`Resource::arn:<AWS::Partition>:kms:<AWS::Region>:<AWS::AccountId>:key/*`],
      },
      {
        id: 'AwsSolutions-IAM5',
        reason: 'SSM parameter path wildcard scoped to /vla-simulator/* prefix for HF token and other simulator secrets.',
        appliesTo: [`Resource::arn:<AWS::Partition>:ssm:<AWS::Region>:<AWS::AccountId>:parameter/vla-simulator/*`],
      },
    ]);

    NagSuppressions.addResourceSuppressionsByPath(this, `/${id}/${roleLogicalId}/DefaultPolicy/Resource`, [
      {
        id: 'AwsSolutions-IAM5',
        reason: 'CDK auto-generates this policy for S3 Asset download from CDK bootstrap bucket. Required by s3assets.Asset.grantRead().',
        appliesTo: [
          'Action::s3:GetBucket*',
          'Action::s3:GetObject*',
          'Action::s3:List*',
          { regex: '/Resource::arn:.+:s3:::cdk-hnb659fds-assets-.+/' },
        ],
      },
    ]);
  }
}
