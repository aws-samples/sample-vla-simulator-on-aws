#!/usr/bin/env node
/**
 * vla-simulator CDK App Entry Point
 *
 * 사용법: deploy.py가 자동 호출 (직접 호출 시 예시):
 *   npx cdk deploy GR00T-Demo                  -c region=us-east-1 -c vla=gr00t
 *   npx cdk deploy GR00T-GR1-Demo              -c region=us-east-1 -c vla=gr00t-gr1
 *   npx cdk deploy Pi-Demo                     -c region=us-east-1 -c vla=pi
 *   npx cdk deploy OpenVLA-OFT-Demo            -c region=us-east-1 -c vla=openvla-oft
 *   npx cdk deploy OpenVLA-OFT-Spatial-Demo    -c region=us-east-1 -c vla=openvla-oft -c libero_suite=spatial
 *   npx cdk deploy LAP-Demo                    -c region=us-east-1 -c vla=lap
 */

import * as cdk from 'aws-cdk-lib';
import { Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks } from 'cdk-nag';
import { VlaSimulatorStack } from '../lib/vla-simulator-stack';

const app = new cdk.App();

const region = app.node.tryGetContext('region') ?? 'ap-northeast-2';
const vla: string = app.node.tryGetContext('vla');

if (!vla || !['gr00t', 'gr00t-gr1', 'pi', 'openvla-oft', 'lap'].includes(vla)) {
  throw new Error(
    'CDK context "vla" is required. Pass -c vla=gr00t, -c vla=gr00t-gr1, -c vla=pi, -c vla=openvla-oft, or -c vla=lap.\n' +
    'Use deploy.py which sets this automatically.',
  );
}

// OpenVLA-OFT only: LIBERO suite ("10" = default, legacy stack name preserved)
const OFT_SUITES = ['spatial', 'object', 'goal', '10', 'long'] as const;
const normaliseSuite = (s: string): string => (s === 'long' ? '10' : s);
const rawSuite: string = app.node.tryGetContext('libero_suite') ?? '10';
if (vla === 'openvla-oft' && !OFT_SUITES.includes(rawSuite as typeof OFT_SUITES[number])) {
  throw new Error(
    `Invalid libero_suite "${rawSuite}". Supported: ${OFT_SUITES.join(', ')}.`,
  );
}
const liberoSuite = vla === 'openvla-oft' ? normaliseSuite(rawSuite) : '';

const suiteCap = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
const oftStackName = (suite: string): string =>
  suite === '10' ? 'OpenVLA-OFT-Demo' : `OpenVLA-OFT-${suiteCap(suite)}-Demo`;

const stackNameMap: Record<string, string> = {
  'gr00t':       'GR00T-Demo',
  'gr00t-gr1':   'GR00T-GR1-Demo',
  'pi':          'Pi-Demo',
  'openvla-oft': oftStackName(liberoSuite),
  'lap':         'LAP-Demo',
};
const stackName = stackNameMap[vla];

const oftDescription = (suite: string): string => {
  if (suite === '10') return 'OpenVLA-OFT + LIBERO-10 (Franka Panda, long-horizon)';
  return `OpenVLA-OFT + LIBERO-${suiteCap(suite)} (Franka Panda)`;
};

const descriptionMap: Record<string, string> = {
  'gr00t':       'GR00T N1.7 + LIBERO (Franka Panda)',
  'gr00t-gr1':   'GR00T N1.6 + RoboCasa (Fourier GR1 humanoid)',
  'pi':          'π0.5',
  'openvla-oft': oftDescription(liberoSuite),
  'lap':         'LAP-3B + LIBERO-Spatial (Franka Panda, JAX policy server + sim client)',
};

const stack = new VlaSimulatorStack(app, stackName, {
  vla,
  liberoSuite,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `vla-simulator ${descriptionMap[vla]} fire-and-forget simulation pipeline`,
});

Aspects.of(stack).add(new AwsSolutionsChecks({ verbose: true }));
