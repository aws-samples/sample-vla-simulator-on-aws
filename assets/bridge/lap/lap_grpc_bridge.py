#!/usr/bin/env python3
"""
lap_grpc_bridge.py — openpi WebSocket ↔ LAP-3B gRPC bridge

LAP의 LIBERO eval(scripts/libero/main.py)이 로컬 LAP policy server(WebSocket:8000)에
접속하는 대신 vla-lap-realtime internal NLB(gRPC:50055)로 추론을 위임한다.

프로토콜 흐름:
  LAP libero client (openpi_client.WebsocketClientPolicy ws://localhost:8000)
    → [bridge WebSocket server]
    → gRPC NLB:50055 (LAPInference.Infer)
    → ECS vla-lap-realtime (LAP-3B JAX 모델)

π0.5 bridge(pi_grpc_bridge.py)와 동일 구조 — 둘 다 openpi WebsocketClientPolicy 클라이언트.
LAP 고유 차이점 (lap@3958d146 scripts/libero/main.py obs_to_request 기준):
  - obs dict가 NESTED — "observation" 하위에 base_0_rgb/left_wrist_0_rgb/state.
    (π0.5는 "observation/image" 처럼 slash-flat 키)
  - state 10-dim: eef_pos(3) + eef_rot6d(6) + gripper(1). (π0.5는 8-dim)
  - prompt + frame_description 둘 다 전달 (flow 정책은 빈 frame_description 무시).
  - flow 정책 출력 actions shape = (action_horizon=10, action_dim=7).

WebSocket 프로토콜 (openpi WebsocketPolicyServer 호환):
  1. 연결 수락 → metadata dict 전송 (msgpack)
  2. 루프: obs dict 수신 (msgpack-numpy) → gRPC Infer → action dict 송신 (msgpack-numpy)

gRPC 프로토콜 (lap.proto LAPInference):
  InferRequest  { base_image, wrist_image, instruction, state(10), frame_description }
  InferResponse { actions: float32 bytes shape=(10,7), shape_info: {chunk_length, action_dim} }

obs → InferRequest 변환 (LAP main.py obs_to_request가 보내는 키):
  "observation" : {
      "base_0_rgb"       : (224, 224, 3) uint8 — agentview (third-person)
      "left_wrist_0_rgb" : (224, 224, 3) uint8 — robot eye-in-hand
      "state"            : (10,) float32 — eef_pos(3)+eef_rot6d(6)+gripper(1)
  }
  "prompt"            : str — task description
  "frame_description" : str — CoT 프레임 힌트

InferResponse → action dict 변환:
  actions bytes → np.frombuffer → (chunk_length, action_dim) → {"actions": (H, 7)}

사용법:
  pip install grpcio grpcio-tools msgpack-numpy websockets Pillow numpy
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. lap.proto
  python lap_grpc_bridge.py --endpoint NLB_HOST:50055 [--port 8000]
"""

import argparse
import asyncio
import io
import logging
import time

import msgpack
import numpy as np
from PIL import Image
import websockets.asyncio.server as _ws_server

import grpc
import lap_pb2
import lap_pb2_grpc


def _unpack_ndarray(obj):
    """Unpack msgpack-encoded numpy array dict (openpi msgpack_numpy format)."""
    if isinstance(obj, dict):
        k = b"__ndarray__" if b"__ndarray__" in obj else ("__ndarray__" if "__ndarray__" in obj else None)
        if k is not None:
            data_k  = b"data"  if b"data"  in obj else "data"
            dtype_k = b"dtype" if b"dtype" in obj else "dtype"
            shape_k = b"shape" if b"shape" in obj else "shape"
            data  = obj[data_k]
            dtype = obj[dtype_k]
            shape = obj[shape_k]
            if isinstance(dtype, bytes):
                dtype = dtype.decode()
            return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=tuple(shape))
    return obj


def _decode_key(k):
    return k.decode("utf-8") if isinstance(k, bytes) else k


def _normalize_obs(raw_obs: dict) -> dict:
    """Normalize obs dict: decode bytes keys to str, unpack nested __ndarray__ dicts.

    LAP obs는 한 단계 nested — top-level "observation"이 dict라서 그 안의 키도 재귀 정규화.
    """
    obs = {}
    for k, v in raw_obs.items():
        key = _decode_key(k)
        if key == "observation" and isinstance(v, dict):
            # nested observation dict: decode keys + unpack each ndarray
            inner = {}
            for ik, iv in v.items():
                inner[_decode_key(ik)] = _unpack_ndarray(iv)
            obs[key] = inner
        else:
            obs[key] = _unpack_ndarray(v)
    return obs


logging.basicConfig(
    level=logging.INFO,
    format="[lap-bridge] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# LAP lap_libero 고정값 (vla-hub serve.py / lap.proto): action_dim=7, action_horizon=10.
STATE_DIM = 10
ACTION_DIM = 7
ACTION_HORIZON = 10


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image(arr: np.ndarray) -> bytes:
    """numpy (H, W, C) uint8 → JPEG bytes"""
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _zeros_image() -> bytes:
    """blank 224×224 JPEG (image absent fallback)"""
    return _encode_image(np.zeros((224, 224, 3), dtype=np.uint8))


def _to_image_bytes(raw) -> bytes:
    if raw is None:
        return _zeros_image()
    arr = np.asarray(raw, dtype=np.uint8)
    while arr.ndim > 3:
        arr = arr[0]
    return _encode_image(arr)


# ── obs → InferRequest ────────────────────────────────────────────────────────

def _obs_to_infer_request(obs: dict) -> lap_pb2.InferRequest:
    """
    LAP LIBERO client obs dict → gRPC InferRequest

    lap scripts/libero/main.py obs_to_request() sends (NESTED):
      "observation" : {
          "base_0_rgb"       : (224, 224, 3) uint8
          "left_wrist_0_rgb" : (224, 224, 3) uint8
          "state"            : (10,) float32 — eef_pos(3)+eef_rot6d(6)+gripper(1)
      }
      "prompt"            : str
      "frame_description" : str
    """
    obs = _normalize_obs(obs)
    inner = obs.get("observation", {})
    if not isinstance(inner, dict):
        log.warning("obs['observation'] is not a dict (got %s) — using empty", type(inner))
        inner = {}

    # ── images ────────────────────────────────────────────────────────────────
    base_raw = inner.get("base_0_rgb")
    if base_raw is None:
        log.warning("obs key 'observation/base_0_rgb' not found — using zeros. obs keys: %s, inner keys: %s",
                    list(obs.keys()), list(inner.keys()))
    base_bytes  = _to_image_bytes(base_raw)
    wrist_bytes = _to_image_bytes(inner.get("left_wrist_0_rgb"))

    # ── instruction ─────────────────────────────────────────────────────────
    prompt = obs.get("prompt", "")
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    elif not isinstance(prompt, str):
        prompt = str(prompt)

    # ── frame_description ─────────────────────────────────────────────────────
    frame_desc = obs.get("frame_description", "")
    if isinstance(frame_desc, bytes):
        frame_desc = frame_desc.decode("utf-8")
    elif not isinstance(frame_desc, str):
        frame_desc = str(frame_desc)

    # ── state (10,): eef_pos(3) + eef_rot6d(6) + gripper(1) ───────────────────
    state_raw = inner.get("state")
    if state_raw is not None:
        state = np.asarray(state_raw, dtype=np.float32).flatten()[:STATE_DIM]
    else:
        state = np.zeros(STATE_DIM, dtype=np.float32)
    if len(state) < STATE_DIM:
        state = np.pad(state, (0, STATE_DIM - len(state)))

    return lap_pb2.InferRequest(
        base_image=base_bytes,
        wrist_image=wrist_bytes,
        instruction=prompt,
        state=state.tolist(),
        frame_description=frame_desc,
    )


# ── InferResponse → action dict ───────────────────────────────────────────────

def _infer_response_to_action(resp: lap_pb2.InferResponse) -> dict:
    """
    gRPC InferResponse.actions → openpi action dict

    actions bytes → np.frombuffer → reshape using shape_info (fallback action_dim=7).
    flow 정책: (action_horizon=10, action_dim=7).
    """
    raw = np.frombuffer(resp.actions, dtype=np.float32).copy()
    # Prefer shape_info from the server; fall back to ACTION_DIM column inference.
    shape_info = dict(resp.shape_info) if resp.shape_info else {}
    chunk_length = int(shape_info.get("chunk_length", 0))
    action_dim   = int(shape_info.get("action_dim", 0))

    if chunk_length > 0 and action_dim > 0 and chunk_length * action_dim == raw.size:
        actions = raw.reshape(chunk_length, action_dim)
    elif raw.size % ACTION_DIM == 0:
        actions = raw.reshape(-1, ACTION_DIM)
    else:
        log.warning("Unexpected actions length %d (shape_info=%s) — using raw 1D", raw.size, shape_info)
        actions = raw

    return {"actions": actions}


# ── WebSocket handler ─────────────────────────────────────────────────────────

def _openpi_pack_array(obj):
    """openpi msgpack_numpy.pack_array — __ndarray__ format."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    return obj


def _zeros_action() -> dict:
    return {"actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)}


def _make_handler(stub: "lap_pb2_grpc.LAPInferenceStub", grpc_timeout: float):
    packer = msgpack.Packer(default=_openpi_pack_array)

    async def handler(websocket: _ws_server.ServerConnection):
        log.info("Client connected: %s", websocket.remote_address)
        req_count = 0

        # openpi WebsocketPolicyServer sends metadata on connect
        metadata = {"bridge": "lap_grpc_bridge", "nlb": True}
        await websocket.send(packer.pack(metadata))

        try:
            async for raw in websocket:
                t0 = time.monotonic()

                # openpi uses its own __ndarray__ format (not PyPI msgpack-numpy);
                # _normalize_obs handles the conversion.
                try:
                    obs = msgpack.unpackb(raw, raw=False)
                except Exception as exc:
                    log.error("msgpack decode failed: %s", exc)
                    continue

                # obs → gRPC InferRequest
                try:
                    infer_req = _obs_to_infer_request(obs)
                except Exception as exc:
                    log.error("obs→InferRequest failed: %s", exc)
                    await websocket.send(packer.pack(_zeros_action()))
                    continue

                # gRPC call
                try:
                    infer_resp = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda req=infer_req: stub.Infer(req, timeout=grpc_timeout),
                    )
                except grpc.RpcError as exc:
                    log.error("gRPC Infer failed: %s %s", exc.code(), exc.details())
                    await websocket.send(packer.pack(_zeros_action()))
                    continue

                # response → action dict
                try:
                    action = _infer_response_to_action(infer_resp)
                except Exception as exc:
                    log.error("InferResponse decode failed: %s", exc)
                    action = _zeros_action()

                infer_ms = (time.monotonic() - t0) * 1000
                action["server_timing"] = {"infer_ms": infer_ms}

                await websocket.send(packer.pack(action))

                req_count += 1
                if req_count % 20 == 0:
                    log.info("Processed %d infer requests (last: %.1f ms)", req_count, infer_ms)

        except Exception as exc:
            log.info("Connection closed: %s (%s)", websocket.remote_address, exc)

        log.info("Client disconnected: %s (total requests: %d)", websocket.remote_address, req_count)

    return handler


# ── Health check (GET /healthz → 200 OK) ─────────────────────────────────────

import http


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(endpoint: str, ws_port: int, grpc_timeout: float) -> None:
    # Strip grpc:// prefix
    host = endpoint[len("grpc://"):] if endpoint.startswith("grpc://") else endpoint

    channel = grpc.insecure_channel(
        host,
        options=[
            ("grpc.keepalive_time_ms",     30_000),
            ("grpc.keepalive_timeout_ms",  10_000),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    stub = lap_pb2_grpc.LAPInferenceStub(channel)
    log.info("gRPC channel → %s", host)

    # Verify gRPC connectivity (Health RPC)
    log.info("Checking gRPC Health RPC...")
    for attempt in range(10):
        try:
            health_resp = stub.Health(lap_pb2.HealthRequest(), timeout=10.0)
            if health_resp.model_loaded:
                log.info("LAP-3B model loaded and ready (healthy=%s, model_loaded=%s)",
                         health_resp.healthy, health_resp.model_loaded)
                break
            else:
                log.info("Server healthy but model not loaded yet — waiting... (%d/10)", attempt + 1)
                await asyncio.sleep(30)
        except grpc.RpcError as exc:
            log.warning("Health check failed (%d/10): %s — retrying in 30s", attempt + 1, exc.details())
            await asyncio.sleep(30)
    else:
        log.error("gRPC endpoint not healthy after 10 attempts — starting bridge anyway")

    handler = _make_handler(stub, grpc_timeout)

    log.info("WebSocket bridge listening on :%d", ws_port)
    async with _ws_server.serve(
        handler,
        "0.0.0.0",
        ws_port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="openpi WebSocket ↔ LAP-3B gRPC bridge"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="gRPC endpoint (e.g. NLB_HOST:50055 or grpc://NLB_HOST:50055)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket listen port (default: 8000 — openpi policy server default)",
    )
    parser.add_argument(
        "--grpc-timeout",
        type=float,
        default=15.0,
        help="gRPC Infer call timeout in seconds (default: 15.0)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.endpoint, args.port, args.grpc_timeout))


if __name__ == "__main__":
    main()
