# VLA Simulator — 1-Click VLA Simulation on AWS

Run Vision-Language-Action (VLA) robot simulation workloads on AWS GPU instances with a single command. Supports NVIDIA GR00T N1.7, GR00T N1.6 (GR1 humanoid), π0.5 (openpi), OpenVLA-OFT, and LAP-3B.

> See [Showcase — VLA Rollouts](#showcase--vla-rollouts) for sample rollouts across all four policies and a range of LIBERO / RoboCasa verbs.

## Overview

| Feature | Detail |
|---------|--------|
| **Models** | GR00T N1.7-LIBERO, GR00T N1.6-3B (GR1), π0.5 (pi05_libero), OpenVLA-OFT (LIBERO-10), LAP-3B (LIBERO-Spatial) |
| **Simulation** | LIBERO / RoboCasa (robosuite + MuJoCo, headless EGL) |
| **Deploy** | AWS CDK + EC2 GPU (g6/g5, us-east-1) |
| **Results** | S3 (MP4 video + summary) + SNS email |
| **Cleanup** | Auto-terminate EC2; run `destroy.py` for stack teardown |

### Supported VLA Combinations

| `--vla` | `--libero-suite` | Model | Sim Environment | Robot | Stack |
|---------|------------------|-------|----------------|-------|-------|
| `gr00t` | — | GR00T N1.7-LIBERO | LIBERO-10 kitchen tasks | Franka Panda (7-DOF) | `GR00T-Demo` |
| `gr00t-gr1` | — | GR00T N1.6-3B | RoboCasa GR1 tabletop tasks | Fourier GR1 humanoid (22-DOF) | `GR00T-GR1-Demo` |
| `pi` | — | π0.5 (pi05_libero) | LIBERO spatial/object | Franka Panda (7-DOF) | `Pi-Demo` |
| `openvla-oft` | `spatial` | OpenVLA-OFT-7B (LIBERO-Spatial fine-tune) | LIBERO-Spatial | Franka Panda (7-DOF) | `OpenVLA-OFT-Spatial-Demo` |
| `openvla-oft` | `object` | OpenVLA-OFT-7B (LIBERO-Object fine-tune) | LIBERO-Object | Franka Panda (7-DOF) | `OpenVLA-OFT-Object-Demo` |
| `openvla-oft` | `goal` | OpenVLA-OFT-7B (LIBERO-Goal fine-tune) | LIBERO-Goal | Franka Panda (7-DOF) | `OpenVLA-OFT-Goal-Demo` |
| `openvla-oft` | `10` (default, alias `long`) | OpenVLA-OFT-7B (LIBERO-10 fine-tune) | LIBERO-10 long-horizon | Franka Panda (7-DOF) | `OpenVLA-OFT-Demo` |
| `lap` | — | LAP-3B (PaliGemma-3B + Flow Matching, JAX) | LIBERO-Spatial | Franka Panda (7-DOF) | `LAP-Demo` |

### Showcase — VLA Rollouts

Sample rollouts captured directly from `deploy.py` runs and synced from each stack's S3 results bucket. GIFs are downsampled previews (288 px, 10 fps, ≤8 s); click-through to the full-quality MP4 in [`docs/showcase/`](docs/showcase/) for the original frame rate and resolution.

#### `pi` — π0.5 on LIBERO (Franka Panda 7-DOF)

`libero_spatial = 0.99` (10 tasks × 10 episodes), `libero_object = 0.96` (10 tasks × 5 episodes) — `g5.xlarge`, validated 2026-04-22.

| LIBERO-Spatial — pick black bowl between plate and ramekin | LIBERO-Object — pick milk and place in basket |
|---|---|
| ![π0.5 spatial — between plate and ramekin (success)](docs/showcase/pi/libero-spatial-between-plate-ramekin-success.gif) | ![π0.5 object — milk in basket (success)](docs/showcase/pi/libero-object-milk-success.gif) |
| MP4: [`pi/libero-spatial-between-plate-ramekin-success.mp4`](docs/showcase/pi/libero-spatial-between-plate-ramekin-success.mp4) | MP4: [`pi/libero-object-milk-success.mp4`](docs/showcase/pi/libero-object-milk-success.mp4) |

#### `openvla-oft` — OpenVLA-OFT on LIBERO-10 long-horizon

Each task is a two-stage instruction. Long-horizon means the policy must complete one sub-goal, recognise it, then proceed to the next. Verb diversity below shows the same checkpoint following structurally different instructions. `g6.xlarge`, validated 2026-05-04.

| `put both the alphabet soup and the tomato sauce in the basket` | `turn on the stove and put the moka pot on it` |
|---|---|
| ![OpenVLA-OFT — soup + sauce (success)](docs/showcase/openvla-oft/libero-10-soup-and-sauce-success.gif) | ![OpenVLA-OFT — stove + moka pot (success)](docs/showcase/openvla-oft/libero-10-stove-moka-pot-success.gif) |
| MP4: [`openvla-oft/libero-10-soup-and-sauce-success.mp4`](docs/showcase/openvla-oft/libero-10-soup-and-sauce-success.mp4) | MP4: [`openvla-oft/libero-10-stove-moka-pot-success.mp4`](docs/showcase/openvla-oft/libero-10-stove-moka-pot-success.mp4) |

| `put the white mug on the left plate and the yellow mug on the right` | `pick up the book and place it in the back compartment` |
|---|---|
| ![OpenVLA-OFT — mug placement (success)](docs/showcase/openvla-oft/libero-10-mug-left-yellow-right-success.gif) | ![OpenVLA-OFT — book in compartment (success)](docs/showcase/openvla-oft/libero-10-book-compartment-success.gif) |
| MP4: [`openvla-oft/libero-10-mug-left-yellow-right-success.mp4`](docs/showcase/openvla-oft/libero-10-mug-left-yellow-right-success.mp4) | MP4: [`openvla-oft/libero-10-book-compartment-success.mp4`](docs/showcase/openvla-oft/libero-10-book-compartment-success.mp4) |

#### `lap` — LAP-3B on LIBERO-Spatial

`libero_spatial = 0.98` (10 tasks × 5 trials, `g6.xlarge`, validated 2026-05-17). Same scene as `pi` above, different policy — useful for side-by-side comparison. The `cookie_box` task is the one consistent failure mode (paper Table III range 85–95% leaves 1–2 tasks expected to miss).

| Success — `pick up the black bowl between the plate and the ramekin` | Failure — `pick up the black bowl on the cookie box` |
|---|---|
| ![LAP-3B — between plate and ramekin (success)](docs/showcase/lap/libero-spatial-between-plate-ramekin-success.gif) | ![LAP-3B — cookie box (failure)](docs/showcase/lap/libero-spatial-cookie-box-failure.gif) |
| MP4: [`lap/libero-spatial-between-plate-ramekin-success.mp4`](docs/showcase/lap/libero-spatial-between-plate-ramekin-success.mp4) | MP4: [`lap/libero-spatial-cookie-box-failure.mp4`](docs/showcase/lap/libero-spatial-cookie-box-failure.mp4) |

#### `gr00t-gr1` — GR00T N1.6 on RoboCasa GR1 humanoid (22-DOF)

The GR1 humanoid is a different embodiment from Franka Panda — two arms, waist, Fourier dexterous hands. `PosttrainPnPNovelFromPlateToBowlSplitA` is in distribution for the post-trained N1.6 (~80% success), while `PnPCanToDrawerClose` is **not supported** by the pre-trained checkpoint and consistently fails — included to make the embodiment-coverage limitation visible. `g6.12xlarge`, validated 2026-04-10.

| Success — `PosttrainPnPNovelFromPlateToBowlSplitA` (in-distribution) | Failure — `PnPCanToDrawerClose` (not supported by N1.6) |
|---|---|
| ![GR00T-GR1 — plate to bowl (success)](docs/showcase/gr00t-gr1/posttrain-pnp-plate-to-bowl-success.gif) | ![GR00T-GR1 — can to drawer (failure, embodiment OOD)](docs/showcase/gr00t-gr1/pnp-can-to-drawer-failure.gif) |
| MP4: [`gr00t-gr1/posttrain-pnp-plate-to-bowl-success.mp4`](docs/showcase/gr00t-gr1/posttrain-pnp-plate-to-bowl-success.mp4) | MP4: [`gr00t-gr1/pnp-can-to-drawer-failure.mp4`](docs/showcase/gr00t-gr1/pnp-can-to-drawer-failure.mp4) |

#### `gr00t` — GR00T N1.7 on LIBERO-10 (Franka Panda)

*Video capture pending — KITCHEN_SCENE3/4 success rate is `1.0` on the validated runs (see [Expected Results](#expected-results)), but those rollout videos were not retained locally. Will backfill on the next `--vla gr00t` deploy.*

---

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

# OpenVLA-OFT — LIBERO-10 long-horizon (~3 hrs, local mode only; default suite)
python deploy.py --vla openvla-oft --email you@example.com

# OpenVLA-OFT — LIBERO-Spatial short-horizon (~1.5 hrs)
python deploy.py --vla openvla-oft --libero-suite spatial --email you@example.com

# Other short-horizon suites: object, goal
python deploy.py --vla openvla-oft --libero-suite object --email you@example.com
python deploy.py --vla openvla-oft --libero-suite goal   --email you@example.com

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
python destroy.py --vla openvla-oft                         # default suite (10)
python destroy.py --vla openvla-oft --libero-suite spatial  # non-default suite
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
| `openvla-oft --libero-suite spatial` | libero_spatial | 97.6% (paper Table I) | validated 2026-06-01 — 0.92 (46/50, 5 trials/task × 10 tasks, g6.xlarge); within paper ±5%p band at n=50 |
| `openvla-oft --libero-suite object` | libero_object | 98.4% (paper Table I) | pending smoke test |
| `openvla-oft --libero-suite goal` | libero_goal | 97.9% (paper Table I) | pending smoke test |
| `openvla-oft --libero-suite 10` | libero_10 (long-horizon) | 94.5% (paper Table I) | validated 2026-05-04 |
| `lap` | libero_spatial | ~85-95% (paper Table III, LIBERO fine-tuned) | validated 2026-05-17 — 0.98 @ 5 trials/task |

**Validated results (us-east-1, g6.12xlarge / g5.xlarge / g6.xlarge):**
- GR00T N1.7: KITCHEN_SCENE3 = 1.0 (5/5), KITCHEN_SCENE4 = 1.0 (3/3)
- GR00T N1.6 + GR1: PosttrainPnP = 0.8 (4/5), PnPCanToDrawer = 0.0 (pre-trained model limitation)
- π0.5: libero_object = 0.94 (47/50)
- OpenVLA-OFT: libero_10 = 1.0 (10/10 at 1 trial/task, g6.xlarge)
- OpenVLA-OFT: libero_spatial = 0.92 (46/50, 5 trials/task × 10 tasks, g6.xlarge) — paper Table I = 97.6%; the gap (5.6%p) is within sampling noise at n=50 (SE ≈ 3.8%p). 8/10 tasks at 5/5; misses concentrated on `on_the_ramekin` (2/5) and `next_to_the_plate` (4/5)
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
├── docs/
│   └── showcase/             # Sample rollout MP4 + GIF preview per VLA policy
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
│       ├── pi/               # gRPC bridge for π0.5
│       └── lap/              # WebSocket↔gRPC bridge for LAP-3B (port 50055)
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

### LAP-3B bridge

LAP shares π0.5's openpi WebSocket bridge pattern. The instance runs only the LIBERO
sim + a WebSocket↔gRPC bridge (`assets/bridge/lap/`); the LAP-3B JAX model runs on the
remote vla-hub LAP ECS task (port 50055). No local checkpoint download.

```bash
# Set in models/lap.yaml (or use ssm: for nlb_endpoint):
#   bridge.vpc_id: vpc-xxxxxxxxxxxxxxxxx                       # the vla-hub VPC
#   bridge.nlb_endpoint: internal-xxx.elb.us-west-2.amazonaws.com:50055
#                        or ssm:/vla-hub/lap/3b/grpc-endpoint

python deploy.py --vla lap --bridge
```

> Bridge differs from π0.5 only in the wire contract: LAP uses a nested observation
> (`observation.base_0_rgb` / `left_wrist_0_rgb` / `state`), a 10-dim state
> (eef_pos 3 + eef_rot6d 6 + gripper 1), a `frame_description` field, and a (10,7)
> action chunk — see `assets/bridge/lap/lap.proto`.

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
