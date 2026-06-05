import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

function makeApp(vla: string, extraCtx: Record<string, string> = {}) {
  const app = new cdk.App({
    context: { region: 'us-east-1', vla, notify_email: 'test@example.com', ...extraCtx },
  });
  const stackName = vla === 'gr00t' ? 'GR00T-Demo' : 'Pi-Demo';
  return new VlaSimulatorStack(app, stackName, {
    vla,
    env: { account: '123456789012', region: 'us-east-1' },
  });
}

test('GR00T stack has EC2 instance', () => {
  const stack = makeApp('gr00t');
  const template = Template.fromStack(stack);
  template.resourceCountIs('AWS::EC2::Instance', 1);
});

test('Pi stack has EC2 instance', () => {
  const stack = makeApp('pi');
  const template = Template.fromStack(stack);
  template.resourceCountIs('AWS::EC2::Instance', 1);
});

test('Both stacks have S3 results bucket', () => {
  for (const vla of ['gr00t', 'pi']) {
    const stack = makeApp(vla);
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::S3::Bucket', 1);
  }
});

test('Both stacks have SNS topic', () => {
  for (const vla of ['gr00t', 'pi']) {
    const stack = makeApp(vla);
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::SNS::Topic', 1);
  }
});
