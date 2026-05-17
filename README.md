# VLA Simulator — 1-Click VLA Simulation on AWS

Run Vision-Language-Action (VLA) robot simulation workloads on AWS GPU instances with a single command. Supports NVIDIA GR00T N1.7, GR00T N1.6 (GR1 humanoid), π0.5 (openpi), OpenVLA-OFT, and LAP-3B.

## Overview

| Feature | Detail |
|---------|--------|
| **Models** | GR00T N1.7-LIBERO, GR00T N1.6-3B (GR1), π0.5 (pi05_libero), OpenVLA-OFT (LIBERO-10), LAP-3B (LIBERO-Spatial) |
| **Simulation** | LIBERO / RoboCasa (robosuite + MuJoCo, headless EGL) |
| **Deploy** | AWS CDK + EC2 GPU (g6/g5, us-east-1) |
| **Results** | S3 (MP4 video + summary) + SNS email |
| **Cleanup** | Auto-terminate EC2; run `destroy.py` for stack teardown |

### Supported VLA Combinations

| `--vla` | Model | Sim Environment | Robot | Stack |
|---------|-------|----------------|-------|-------|
| `gr00t` | GR00T N1.7-LIBERO | LIBERO-10 kitchen tasks | Franka Panda (7-DOF) | `GR00T-Demo` |
| `gr00t-gr1` | GR00T N1.6-3B | RoboCasa GR1 tabletop tasks | Fourier GR1 humanoid (22-DOF) | `GR00T-GR1-Demo` |
| `pi` | π0.5 (pi05_libero) | LIBERO spatial/object | Franka Panda (7-DOF) | `Pi-Demo` |
| `openvla-oft` | OpenVLA-OFT-7B (LIBERO-10 fine-tune) | LIBERO-10 long-horizon | Franka Panda (7-DOF) | `OpenVLA-OFT-Demo` |
| `lap` | LAP-3B (PaliGemma-3B + Flow Matching, JAX) | LIBERO-Spatial | Franka Panda (7-DOF) | `LAP-Demo` |

### Architecture

```
deploy.py
  │
  ├─ generate.py          → assets/userdata/{vla}.sh
  │
  └─ CDK deploy
       ├─ VPC + SG
       ├─ S3 ResultsBucket (RETAIN)
       ├─ SNS + EmailSubscription
       ├─ IAM Role
       ├─ AzSelector Lambda  →  EC2 (g6.12xlarge / g5.xlarge)
       └─ WaitCondition      ←  cfn_signal from UserData
```

**Two deployment modes:**

- **Local mode** — model runs directly on the EC2 instance (default)
- **Bridge mode** — EC2 runs simulation env; model inference forwarded to an existing VLA Hub ECS endpoint

---

## Prerequisites

```bash
# Python deps
pip install -r requirements.txt   # boto3, pyyaml

# Node / CDK
npm install -g aws-cdk
cd cdk && npm install && cd ..

# AWS credentials (profile or env vars)
aws configure   # or: export AWS_PROFILE=...
aws sts get-caller-identity
```

Supported regions: `us-east-1`, `us-west-2`, `ap-northeast-1`, `ap-northeast-2`, `eu-central-1`

---

## Quick Start

### 1. Configure

Edit `simulator-config.yaml`:

```yaml
deployment:
  region: us-east-1
  notify_email: you@example.com
```

### 2. Deploy

```bash
# GR00T N1.7 — LIBERO kitchen tasks (~120 min)
python deploy.py --vla gr00t --email you@example.com

# GR00T N1.6 + GR1 humanoid — RoboCasa tabletop tasks (~90 min)
python deploy.py --vla gr00t-gr1 --email you@example.com

# π0.5 — LIBERO spatial + object tasks (~4 hrs)
python deploy.py --vla pi --email you@example.com

# OpenVLA-OFT — LIBERO-10 long-horizon (~3 hrs, local mode only)
python deploy.py --vla openvla-oft --email you@example.com

# LAP-3B — LIBERO-Spatial (zero-shot cross-embodiment VLA, JAX policy server, ~1.5-2.5 hrs)
python deploy.py --vla lap --email you@example.com
```

On first deploy you will receive an SNS subscription confirmation email — **click the link** to enable notifications.

### 3. Monitor (optional)

```bash
# GR00T N1.7 logs
aws logs tail /gr00t/userdata --follow --region us-east-1

# GR00T N1.6 + GR1 logs
aws logs tail /gr00t-gr1/userdata --follow --region us-east-1

# π0.5 logs
aws logs tail /pi/userdata --follow --region us-east-1

# OpenVLA-OFT logs
aws logs tail /openvla-oft/userdata --follow --region us-east-1

# LAP-3B logs
aws logs tail /lap/userdata --follow --region us-east-1
```

### 4. Results

When simulation finishes you receive an SNS email with download instructions:

```bash
# GR00T results
aws s3 sync s3://vla-sim-results-gr00t-demo-us-east-1-<ACCOUNT>/RUN_ID/ ./results/ --region us-east-1

# π0.5 results
aws s3 sync s3://vla-sim-results-pi-demo-us-east-1-<ACCOUNT>/RUN_ID/ ./results/ --region us-east-1
```

Each `task-N/` folder contains:
- `videos/rollout_<task_name>_success.mp4` — successful episodes
- `videos/rollout_<task_name>_failure.mp4` — failed episodes
- `videos/<name>.txt` — per-video metadata
- `summary.txt` — `success_rate`, `suite`, `num_episodes_per_task`
- `compose.log` / `userdata.log` — execution logs

### 5. Cleanup

```bash
python destroy.py --vla gr00t
python destroy.py --vla gr00t-gr1
python destroy.py --vla pi
python destroy.py --vla openvla-oft
python destroy.py --vla lap
```

The S3 results bucket is **retained** after stack deletion to preserve simulation outputs.

---

## Expected Results

| `--vla` | Task | Expected Success Rate | Source |
|---------|------|-----------------------|--------|
| `gr00t` | KITCHEN_SCENE3 (stove + moka pot) | ~90–100% | validated 2026-04-27 |
| `gr00t` | KITCHEN_SCENE4 (black bowl in drawer) | ~90–100% | validated 2026-04-27 |
| `gr00t-gr1` | PosttrainPnPNovelFromPlateToBowlSplitA (GR1) | ~80% | validated 2026-04-12 |
| `gr00t-gr1` | PnPCanToDrawerClose (GR1) | 0% (pre-trained N1.6 not supported) | validated 2026-04-12 |
| `pi` | libero_object | ~80–94% | validated 2026-04-27 |
| `pi` | libero_spatial | ~85–95% | validated 2026-04-27 |
| `openvla-oft` | libero_10 (long-horizon) | ~94.5% (paper) | validated 2026-05-04 |
| `lap` | libero_spatial | ~85-95% (paper Table III, LIBERO fine-tuned) | validated 2026-05-17 — 0.98 @ 5 trials/task |

**Validated results (us-east-1, g6.12xlarge / g5.xlarge / g6.xlarge):**
- GR00T N1.7: KITCHEN_SCENE3 = 1.0 (5/5), KITCHEN_SCENE4 = 1.0 (3/3)
- GR00T N1.6 + GR1: PosttrainPnP = 0.8 (4/5), PnPCanToDrawer = 0.0 (pre-trained model limitation)
- π0.5: libero_object = 0.94 (47/50)
- OpenVLA-OFT: libero_10 = 1.0 (10/10 at 1 trial/task, g6.xlarge)
- LAP-3B: libero_spatial = 0.98 (49/50, 5 trials/task × 10 tasks, g6.xlarge) — exceeds paper Table III range (85–95%); requires upstream `scripts/libero/main.py` vertical-flip patch (see `templates/lap-userdata.sh.j2`)

---

## Configuration Reference

### `simulator-config.yaml` (shared)

| Key | Default | Description |
|-----|---------|-------------|
| `deployment.region` | `us-east-1` | AWS region |
| `deployment.notify_email` | *(required)* | SNS notification email |
| `deployment.s3_results_prefix` | `vla-sim-results` | S3 bucket name prefix |
| `deployment.auto_terminate` | `true` | Auto-terminate EC2 after completion |

### `models/gr00t.yaml`

| Key | Description |
|-----|-------------|
| `model.hf_repo` | HuggingFace repo for checkpoint download |
| `model.hf_subfolder` | Subfolder within repo (e.g. `libero_10`) |
| `model.hf_model_revision` | Pinned commit SHA for reproducibility |
| `model.isaac_groot_commit` | Pinned Isaac-GR00T commit SHA |
| `instance.preferred` | GPU instance type fallback list |
| `instance.ebs_gb` | Root EBS volume size (GB) |
| `tasks` | List of LIBERO tasks to evaluate |

### `models/pi.yaml`

| Key | Description |
|-----|-------------|
| `model.openpi_commit` | Pinned openpi commit SHA |
| `instance.preferred` | GPU instance type fallback list |
| `tasks[].suite` | LIBERO suite name (`libero_spatial`, `libero_object`, etc.) |
| `tasks[].num_episodes` | Episodes per task |

---

## Project Structure

```
vla-simulator/
├── simulator-config.yaml     # Shared deployment settings
├── models/
│   ├── gr00t.yaml            # GR00T N1.7 config
│   ├── gr00t-gr1.yaml        # GR00T N1.6 + GR1 humanoid config
│   ├── pi.yaml               # π0.5 config
│   ├── openvla-oft.yaml      # OpenVLA-OFT config (per-suite checkpoints)
│   └── lap.yaml              # LAP-3B config
├── deploy.py                 # 1-click deploy entrypoint
├── destroy.py                # Stack teardown
├── generate.py               # Generates assets/userdata/{vla}.sh from templates
├── requirements.txt
├── cdk/
│   ├── bin/app.ts
│   └── lib/
│       ├── vla-simulator-stack.ts   # Unified CDK stack
│       └── az-selector.ts           # GPU capacity Lambda (AZ/type fallback)
├── assets/
│   ├── userdata/             # Generated scripts (gitignored)
│   └── bridge/
│       ├── gr00t/            # ZMQ-gRPC bridge for GR00T
│       └── pi/               # gRPC bridge for π0.5
└── templates/
    ├── gr00t-userdata.sh.j2        # Jinja2 template for GR00T UserData
    ├── gr00t-gr1-userdata.sh.j2    # GR00T N1.6 + GR1 humanoid
    ├── pi-userdata.sh.j2           # π0.5 (Docker Compose)
    ├── openvla-oft-userdata.sh.j2  # OpenVLA-OFT (single conda env)
    └── lap-userdata.sh.j2          # LAP-3B (uv 2-venv: JAX policy + LIBERO sim)
```

---

## Bridge Mode

Bridge mode connects the simulation environment on EC2 to an external VLA model endpoint (e.g. VLA Hub on ECS), without downloading the model locally.

### GR00T bridge

```bash
# Set endpoint in models/gr00t.yaml (or SSM):
#   bridge.remote_grpc_endpoint: ssm:/vla-hub/gr00t/n1-7/grpc-endpoint
#   bridge.vpc_id: ssm:/vla-hub/vpc-id

python deploy.py --vla gr00t --bridge
```

### π0.5 bridge

```bash
# Set in models/pi.yaml:
#   bridge.vpc_id: vpc-xxxxxxxxxxxxxxxxx
#   bridge.nlb_endpoint: internal-xxx.elb.us-east-1.amazonaws.com:50052

python deploy.py --vla pi --bridge
```

---

## Troubleshooting

### "Embodiment tag 'LIBERO_PANDA' is not supported"

The **base** GR00T N1.7-3B model does not support LIBERO simulation. Use the fine-tuned checkpoint:

```yaml
# models/gr00t.yaml
model:
  hf_repo: nvidia/GR00T-N1.7-LIBERO
  hf_subfolder: libero_10
```

### "No visible GPU devices" (π0.5 task-0)

JAX inside the Docker container fails if GPU passthrough isn't ready before the task loop starts. The `[2.5/5]` step in the UserData script polls `docker run --gpus all nvidia-smi` until GPU is confirmed before proceeding.

### Stack rollback / InsufficientInstanceCapacity

The `AzSelectorConstruct` Lambda automatically retries across AZs and falls back through the instance type preference list (`g6.12xlarge` → `g5.12xlarge` → `g6.xlarge` → `g5.xlarge` for GR00T). No manual intervention needed — CDK rollback means no capacity was found; re-deploying will retry.

### CDK synth cdk.out conflict (parallel deploys)

Use model-specific output directories:

```bash
npx cdk deploy GR00T-Demo --output cdk.out-gr00t -c vla=gr00t -c region=us-east-1 ...
npx cdk deploy Pi-Demo    --output cdk.out-pi    -c vla=pi    -c region=us-east-1 ...
```

`deploy.py` handles this automatically.

### apt lock on DLAMI boot

`unattended-upgrades` holds the apt lock for ~2 minutes on fresh boot. The UserData waits up to 120 seconds then forcibly kills the process. No action needed.

---

## Cost Estimate (us-east-1, on-demand)

| Instance | Spot/OD | Hourly | GR00T (~2h) | π0.5 (~4h) | LAP (~2.5h) |
|----------|---------|--------|-------------|------------|-------------|
| g6.12xlarge | On-Demand | ~$4.09 | ~$8.18 | — | — |
| g5.xlarge   | On-Demand | ~$1.01 | — | ~$4.04 | — |
| g6.xlarge   | On-Demand | ~$0.80 | — | — | ~$2.00 |

Use Spot instances via `deploy.py --spot` (not yet implemented) for 60–70% savings.

---

## Related Projects

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA GR00T foundation model
- [openpi](https://github.com/Physical-Intelligence/openpi) — Physical Intelligence π0.5
- [openvla-oft](https://github.com/moojink/openvla-oft) — OpenVLA-OFT (Optimized Fine-Tuning)
- [lap](https://github.com/lihzha/lap) — LAP: Language-Action Pre-Training (zero-shot cross-embodiment, JAX)
- [LIBERO](https://libero-project.github.io/) — Long-horizon robot benchmark
