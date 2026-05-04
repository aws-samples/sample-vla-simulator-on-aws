#!/usr/bin/env node
/**
 * vla-simulator CDK App Entry Point
 *
 * 사용법: deploy.py가 자동 호출 (직접 호출 시 예시):
 *   npx cdk deploy GR00T-Demo        -c region=us-east-1 -c vla=gr00t
 *   npx cdk deploy GR00T-GR1-Demo    -c region=us-east-1 -c vla=gr00t-gr1
 *   npx cdk deploy Pi-Demo           -c region=us-east-1 -c vla=pi
 *   npx cdk deploy OpenVLA-OFT-Demo  -c region=us-east-1 -c vla=openvla-oft
 */

import * as cdk from 'aws-cdk-lib';
import { Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const app = new cdk.App();

const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const vla: string = app.node.tryGetContext('vla');

if (!vla || !['gr00t', 'gr00t-gr1', 'pi', 'openvla-oft'].includes(vla)) {
  throw new Error(
    'CDK context "vla" is required. Pass -c vla=gr00t, -c vla=gr00t-gr1, -c vla=pi, or -c vla=openvla-oft.\n' +
    'Use deploy.py which sets this automatically.',
  );
}

const stackNameMap: Record<string, string> = {
  'gr00t':       'GR00T-Demo',
  'gr00t-gr1':   'GR00T-GR1-Demo',
  'pi':          'Pi-Demo',
  'openvla-oft': 'OpenVLA-OFT-Demo',
};
const stackName = stackNameMap[vla];

const descriptionMap: Record<string, string> = {
  'gr00t':       'GR00T N1.7 + LIBERO (Franka Panda)',
  'gr00t-gr1':   'GR00T N1.6 + RoboCasa (Fourier GR1 humanoid)',
  'pi':          'π0.5',
  'openvla-oft': 'OpenVLA-OFT + LIBERO-10 (Franka Panda, long-horizon)',
};

const stack = new VlaSimulatorStack(app, stackName, {
  vla,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${descriptionMap[vla]} fire-and-forget simulation pipeline`,
});

Aspects.of(stack).add(new AwsSolutionsChecks({ verbose: true }));
