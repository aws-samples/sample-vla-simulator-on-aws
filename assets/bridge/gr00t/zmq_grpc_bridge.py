#!/usr/bin/env python3
"""
zmq_grpc_bridge.py — ZMQ ↔ gRPC bridge for GR00T enablement-pack

enablement-pack EC2가 모델을 로컬 로드하지 않고 gr00t-realtime NLB에 추론을 위임한다.

  rollout_policy.py (ZMQ REQ :8000) → [bridge] → gRPC NLB:50051

ZMQ REP(:8000) : 기존 rollout_policy.py와 연결 (Isaac-GR00T 코드 수정 없음)
gRPC stub      : gr00t-realtime NLB(:50051) 호출

메시지 변환 흐름:
  ZMQ recv  → MsgSerializer { endpoint envelope + obs dict (numpy __ndarray_class__) }
  gRPC send → InferRequest { image_data(JPEG), instruction, joint states }
  gRPC recv → InferResponse { action_chunks map<string, bytes> }
  ZMQ send  → MsgSerializer { [action_dict, {}] } — matches PolicyServer response format

프로토콜 참고: Isaac-GR00T gr00t/policy/server_client.py (MsgSerializer, PolicyServer, PolicyClient)

사용법:
  python zmq_grpc_bridge.py --endpoint NLB_HOST:50051 [--zmq-port 8000]

추가 의존성 (Isaac-GR00T venv 기본 외):
  pip install "grpcio>=1.80.0" Pillow
  (pyzmq, numpy, msgpack은 Isaac-GR00T venv에 이미 포함)
"""

import argparse
import io
import logging
import sys

import msgpack
import numpy as np
import zmq
from PIL import Image

import grpc
import gr00t_pb2
import gr00t_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="[bridge] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 관절별 DOF — action_chunks bytes → numpy (H, DOF) 복원에 사용
_DOF_MAP = {
    "left_arm":   7,
    "right_arm":  7,
    "left_hand":  6,
    "right_hand": 6,
    "waist":      3,
}


# ── MsgSerializer ─────────────────────────────────────────────────────────────
# Isaac-GR00T PolicyClient/PolicyServer와 동일한 직렬화 (server_client.py 참고)
# numpy array: {"__ndarray_class__": True, "as_npy": <bytes>}

class MsgSerializer:
    @staticmethod
    def to_bytes(data) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False)

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"[bridge] Cannot encode type: {type(obj)}")


# ── Image encoding ─────────────────────────────────────────────────────────────

def _encode_image(image_np: np.ndarray) -> bytes:
    """numpy (H, W, C) uint8 → JPEG bytes"""
    buf = io.BytesIO()
    Image.fromarray(image_np).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ── Obs → gRPC InferRequest ───────────────────────────────────────────────────

def _obs_to_infer_request(obs: dict) -> "gr00t_pb2.InferRequest":
    """
    Gymnasium VecEnv obs dict → gRPC InferRequest

    PolicyClient → PolicyServer (Gr00tSimPolicyWrapper) 프로토콜에서
    obs는 Gr00t sim flat format:
      {
        "video.<key>":  np.ndarray(n_envs, T, H, W, C) uint8,
        "state.<key>":  np.ndarray(n_envs, T, DOF),
        "annotation.human.task_description": np.ndarray or [[str]]
      }
    nested format ({"video": {...}}) 도 fallback으로 지원.
    """
    # ── Video ───────────────────────────────────────────────────────────────
    # Flat format: "video.<cam_key>" 우선 시도
    video_dict = {k[len("video."):]: v for k, v in obs.items() if k.startswith("video.")}
    if not video_dict:
        # Nested format fallback
        video_dict = obs.get("video", {})
    if not video_dict:
        raise ValueError("obs missing 'video.*' key (checked flat and nested)")

    image = next(iter(video_dict.values()))
    while image.ndim > 3:
        image = image[0]  # n_envs=0, T=0, ... 순으로 batch dim 제거
    image_bytes = _encode_image(np.asarray(image, dtype=np.uint8))

    # ── Language / instruction ────────────────────────────────────────────────
    # Flat format: "annotation.human.task_description" or "annotation.human.coarse_action"
    instruction = ""
    for lang_key in [
        "annotation.human.task_description",
        "annotation.human.coarse_action",
        "task",
    ]:
        if lang_key in obs:
            raw = obs[lang_key]
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            # Unwrap nested lists until we get a scalar
            while isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            instruction = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            break
    if not instruction:
        # Nested format fallback
        task = obs.get("language", {}).get("task", [[""]])
        if hasattr(task, "tolist"):
            task = task.tolist()
        row = task[0] if task else [""]
        if hasattr(row, "tolist"):
            row = row.tolist()
        raw_instr = row[0] if row else ""
        instruction = raw_instr.decode("utf-8") if isinstance(raw_instr, bytes) else str(raw_instr)

    # ── State: 관절 배열 → float32 리스트 ────────────────────────────────────
    # Flat format: "state.<joint_key>"
    state_flat = {k[len("state."):]: v for k, v in obs.items() if k.startswith("state.")}
    state_nested = obs.get("state", {})
    state = state_flat if state_flat else state_nested

    def _flat_state(key):
        arr = state.get(key)
        if arr is None:
            return []
        return np.asarray(arr, dtype=np.float32).flatten().tolist()

    return gr00t_pb2.InferRequest(
        image_data=image_bytes,
        instruction=instruction,
        left_arm=_flat_state("left_arm"),
        right_arm=_flat_state("right_arm"),
        left_hand=_flat_state("left_hand"),
        right_hand=_flat_state("right_hand"),
        waist=_flat_state("waist"),
    )


# ── gRPC InferResponse → action dict ─────────────────────────────────────────

def _infer_resp_to_action(resp: "gr00t_pb2.InferResponse") -> dict:
    """
    gRPC InferResponse.action_chunks → action dict with shaped numpy arrays

    InferResponse.action_chunks: map<string, bytes> (float32 tobytes())
    복원: np.frombuffer → reshape (B=1, H, DOF)
    이 형식은 PolicyServer(로컬 Gr00tPolicy)가 반환하는 action_dict와 동일해야 함.
    """
    action_dict = {}
    for k, v_bytes in resp.action_chunks.items():
        arr = np.frombuffer(v_bytes, dtype=np.float32).copy()
        dof = _DOF_MAP.get(k)
        if dof and len(arr) % dof == 0:
            arr = arr.reshape(1, len(arr) // dof, dof)  # (B=1, H, DOF)
        # "action." 프리픽스 추가 — Gr00tSimPolicyWrapper 출력 포맷과 동일
        # gr00t_policy.py: return {f"action.{key}": action[key] for key in action}, info
        action_dict[f"action.{k}"] = arr
    return action_dict


# ── Bridge main loop ──────────────────────────────────────────────────────────

def run(endpoint: str, zmq_port: int) -> None:
    # grpc:// prefix 제거
    host = endpoint[len("grpc://"):] if endpoint.startswith("grpc://") else endpoint

    # gRPC channel → NLB
    channel = grpc.insecure_channel(
        host,
        options=[
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
        ],
    )
    stub = gr00t_pb2_grpc.GR00TInferenceStub(channel)
    log.info("gRPC channel → %s", host)

    # ZMQ REP socket — PolicyServer 드롭인 대체
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{zmq_port}")
    log.info("ZMQ REP listening on :%d", zmq_port)

    req_count = 0
    try:
        while True:
            raw = sock.recv()

            # MsgSerializer 디코딩 — PolicyClient가 보내는 포맷
            try:
                request = MsgSerializer.from_bytes(raw)
            except Exception as exc:
                log.error("Request decode failed: %s", exc)
                sock.send(MsgSerializer.to_bytes({"error": str(exc)}))
                continue

            endpoint_name = request.get("endpoint", "get_action")

            # ── 비추론 엔드포인트 ────────────────────────────────────────────
            if endpoint_name == "ping":
                sock.send(MsgSerializer.to_bytes({"status": "ok", "message": "Server is running"}))
                continue
            elif endpoint_name == "reset":
                sock.send(MsgSerializer.to_bytes({}))
                continue
            elif endpoint_name == "get_modality_config":
                sock.send(MsgSerializer.to_bytes({}))
                continue
            elif endpoint_name == "kill":
                sock.send(MsgSerializer.to_bytes({"status": "ok"}))
                log.info("Kill received — bridge exiting.")
                break
            elif endpoint_name != "get_action":
                log.warning("Unknown endpoint: %s", endpoint_name)
                sock.send(MsgSerializer.to_bytes({"error": f"Unknown endpoint: {endpoint_name}"}))
                continue

            # ── get_action ───────────────────────────────────────────────────
            data = request.get("data", {})
            obs = data.get("observation", {})

            # obs → gRPC InferRequest
            try:
                infer_req = _obs_to_infer_request(obs)
            except Exception as exc:
                log.error("obs → InferRequest failed: %s", exc)
                # PolicyServer 포맷으로 빈 응답 반환: [action_dict, info_dict]
                sock.send(MsgSerializer.to_bytes([{}, {}]))
                continue

            # gRPC Infer 호출
            try:
                infer_resp = stub.Infer(infer_req, timeout=10.0)
            except grpc.RpcError as exc:
                log.error("gRPC Infer failed: %s %s", exc.code(), exc.details())
                sock.send(MsgSerializer.to_bytes([{}, {}]))
                continue

            # InferResponse → action dict → MsgSerializer → ZMQ REP
            # 반환 형식: [action_dict, info_dict] (PolicyServer와 동일)
            # PolicyClient._get_action: return tuple(response) → (action_dict, {})
            action_dict = _infer_resp_to_action(infer_resp)
            sock.send(MsgSerializer.to_bytes([action_dict, {}]))

            req_count += 1
            if req_count % 10 == 0:
                log.info("Processed %d inference requests", req_count)

    except KeyboardInterrupt:
        log.info("Bridge stopped.")
    finally:
        sock.close()
        ctx.term()
        channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZMQ ↔ gRPC bridge — GR00T enablement-pack 원격 모드"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="gRPC endpoint (예: NLB_HOST:50051 또는 grpc://NLB_HOST:50051)",
    )
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=8000,
        help="ZMQ REP 리슨 포트 (기본: 8000 — PolicyServer와 동일)",
    )
    args = parser.parse_args()
    run(args.endpoint, args.zmq_port)


if __name__ == "__main__":
    main()
