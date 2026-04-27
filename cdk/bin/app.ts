#!/usr/bin/env node
/**
 * vla-simulator CDK App Entry Point
 *
 * 사용법: deploy.py가 자동 호출 (직접 호출 시 예시):
 *   npx cdk deploy GR00T-Demo -c region=us-east-1 -c vla=gr00t
 *   npx cdk deploy Pi-Demo    -c region=us-east-1 -c vla=pi
 */

import * as cdk from 'aws-cdk-lib';
import { Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const app = new cdk.App();

const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const vla: string = app.node.tryGetContext('vla');

if (!vla || !['gr00t', 'pi'].includes(vla)) {
  throw new Error(
    'CDK context "vla" is required. Pass -c vla=gr00t or -c vla=pi.\n' +
    'Use deploy.py which sets this automatically.',
  );
}

const stackName = vla === 'gr00t' ? 'GR00T-Demo' : 'Pi-Demo';

const stack = new VlaSimulatorStack(app, stackName, {
  vla,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${vla === 'gr00t' ? 'GR00T N1.7' : 'π0.5'} fire-and-forget simulation pipeline`,
});

Aspects.of(stack).add(new AwsSolutionsChecks({ verbose: true }));
