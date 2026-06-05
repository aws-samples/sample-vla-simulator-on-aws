#!/usr/bin/env python3
"""
pi_grpc_bridge.py — openpi WebSocket ↔ π0.5 gRPC bridge

openpi libero eval이 로컬 openpi_server(WebSocket:8000)에 접속하는 대신
vla-pi-realtime internal NLB(gRPC:50051)로 추론을 위임한다.

프로토콜 흐름:
  openpi libero client (WebSocket ws://localhost:8000)
    → [bridge WebSocket server]
    → gRPC NLB:50051 (PIInference.Infer)
    → ECS vla-pi-realtime (π0.5 JAX 모델)

WebSocket 프로토콜 (openpi WebsocketPolicyServer 호환):
  1. 연결 수락 → metadata dict 전송 (msgpack)
  2. 루프: obs dict 수신 (msgpack-numpy) → gRPC Infer → action dict 송신 (msgpack-numpy)

gRPC 프로토콜 (pi.proto PIInference):
  InferRequest  { exterior_image, wrist_image, instruction, state }
  InferResponse { actions: float32 bytes, shape=(15, 8) }

obs → InferRequest 변환:
  openpi examples/libero/main.py obs 키:
    "observation/image"       : (224, 224, 3) uint8 — agentview (exterior)
    "observation/wrist_image" : (224, 224, 3) uint8 — robot eye-in-hand
    "observation/state"       : (8,) float32 — eef_pos(3)+axis_angle(3)+gripper(2)
    "prompt"                  : str — task description
  Note: msgpack default raw=True → keys may be bytes; _normalize_obs() handles both

InferResponse → action dict 변환:
  actions bytes → np.frombuffer → (15, 8) → openpi action format
    "actions": (15, 8) float32

사용법:
  pip install grpcio grpcio-tools msgpack-numpy websockets Pillow numpy
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pi.proto
  python pi_grpc_bridge.py --endpoint NLB_HOST:50051 [--port 8000]
"""

import argparse
import asyncio
import io
import logging
import sys
import time

import msgpack
import numpy as np
from PIL import Image
import websockets.asyncio.server as _ws_server

import grpc
import pi_pb2
import pi_pb2_grpc


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


def _normalize_obs(raw_obs: dict) -> dict:
    """Normalize obs dict: decode bytes keys to str, unpack nested __ndarray__ dicts."""
    obs = {}
    for k, v in raw_obs.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else k
        obs[key] = _unpack_ndarray(v)
    return obs

logging.basicConfig(
    level=logging.INFO,
    format="[pi-bridge] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image(arr: np.ndarray) -> bytes:
    """numpy (H, W, C) uint8 → JPEG bytes"""
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _zeros_image() -> bytes:
    """blank 224×224 JPEG (wrist_image absent fallback)"""
    return _encode_image(np.zeros((224, 224, 3), dtype=np.uint8))


# ── obs → InferRequest ────────────────────────────────────────────────────────

def _obs_to_infer_request(obs: dict) -> pi_pb2.InferRequest:
    """
    openpi LIBERO client obs dict → gRPC InferRequest

    openpi examples/libero/main.py sends:
      "observation/image"       : (224, 224, 3) uint8 — agentview (exterior)
      "observation/wrist_image" : (224, 224, 3) uint8 — robot eye-in-hand
      "observation/state"       : (8,) float32 — eef_pos(3)+axis_angle(3)+gripper(2)
      "prompt"                  : str — task description
    """
    obs = _normalize_obs(obs)

    # ── exterior image ───────────────────────────────────────────────────────
    ext_raw = obs.get("observation/image")
    if ext_raw is not None:
        ext_arr = np.asarray(ext_raw, dtype=np.uint8)
        while ext_arr.ndim > 3:
            ext_arr = ext_arr[0]
        exterior_bytes = _encode_image(ext_arr)
    else:
        log.warning("obs key 'observation/image' not found — using zeros. Keys: %s", list(obs.keys()))
        exterior_bytes = _zeros_image()

    # ── wrist image ──────────────────────────────────────────────────────────
    wrist_raw = obs.get("observation/wrist_image")
    if wrist_raw is not None:
        wrist_arr = np.asarray(wrist_raw, dtype=np.uint8)
        while wrist_arr.ndim > 3:
            wrist_arr = wrist_arr[0]
        wrist_bytes = _encode_image(wrist_arr)
    else:
        wrist_bytes = _zeros_image()

    # ── instruction ──────────────────────────────────────────────────────────
    prompt = obs.get("prompt", "")
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    elif not isinstance(prompt, str):
        prompt = str(prompt)

    # ── state (8,): eef_pos(3) + axis_angle(3) + gripper(2) ─────────────────
    state_raw = obs.get("observation/state")
    if state_raw is not None:
        state = np.asarray(state_raw, dtype=np.float32).flatten()[:8]
    else:
        state = np.zeros(8, dtype=np.float32)

    # pad to 8 if shorter
    if len(state) < 8:
        state = np.pad(state, (0, 8 - len(state)))

    return pi_pb2.InferRequest(
        exterior_image=exterior_bytes,
        wrist_image=wrist_bytes,
        instruction=prompt,
        state=state.tolist(),
    )


# ── InferResponse → action dict ───────────────────────────────────────────────

def _infer_response_to_action(resp: pi_pb2.InferResponse) -> dict:
    """
    gRPC InferResponse.actions → openpi action dict

    pi05_libero: LiberoOutputs already applies [:, :7] server-side
      actions bytes → np.frombuffer → (chunk_length=10, action_dim=7) float32
    pi05_droid (legacy): 8-dim actions — slice [:, :7] for LIBERO Franka compatibility
    """
    raw = np.frombuffer(resp.actions, dtype=np.float32).copy()
    libero_dim = 7
    infer_dim = 8
    if len(raw) % libero_dim == 0:
        actions = raw.reshape(-1, libero_dim)
    elif len(raw) % infer_dim == 0:
        # legacy pi05_droid path: slice last dim
        actions = raw.reshape(-1, infer_dim)[:, :libero_dim]
    else:
        log.warning("Unexpected actions length %d — using raw", len(raw))
        actions = raw

    return {"actions": actions}


# ── WebSocket handler ─────────────────────────────────────────────────────────

def _openpi_pack_array(obj):
    """openpi msgpack_numpy.pack_array — __ndarray__ format (not PyPI msgpack-numpy nd format)."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    return obj


def _make_handler(stub: pi_pb2_grpc.PIInferenceStub, grpc_timeout: float):
    packer = msgpack.Packer(default=_openpi_pack_array)

    async def handler(websocket: _ws_server.ServerConnection):
        log.info("Client connected: %s", websocket.remote_address)
        req_count = 0

        # openpi WebsocketPolicyServer sends metadata on connect
        metadata = {"bridge": "pi_grpc_bridge", "nlb": True}
        await websocket.send(packer.pack(metadata))

        try:
            async for raw in websocket:
                t0 = time.monotonic()

                # Decode obs dict using plain msgpack (openpi uses its own __ndarray__ format,
                # not compatible with PyPI msgpack-numpy; _normalize_obs handles conversion)
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
                    action = {"actions": np.zeros((15, 8), dtype=np.float32)}
                    await websocket.send(packer.pack(action))
                    continue

                # gRPC call
                try:
                    infer_resp = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda req=infer_req: stub.Infer(req, timeout=grpc_timeout),
                    )
                except grpc.RpcError as exc:
                    log.error("gRPC Infer failed: %s %s", exc.code(), exc.details())
                    action = {"actions": np.zeros((15, 8), dtype=np.float32)}
                    await websocket.send(packer.pack(action))
                    continue

                # response → action dict
                try:
                    action = _infer_response_to_action(infer_resp)
                except Exception as exc:
                    log.error("InferResponse decode failed: %s", exc)
                    action = {"actions": np.zeros((15, 8), dtype=np.float32)}

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
    stub = pi_pb2_grpc.PIInferenceStub(channel)
    log.info("gRPC channel → %s", host)

    # Verify gRPC connectivity (Health RPC)
    log.info("Checking gRPC Health RPC...")
    for attempt in range(10):
        try:
            health_resp = stub.Health(pi_pb2.HealthRequest(), timeout=10.0)
            if health_resp.model_loaded:
                log.info("π0.5 model loaded and ready (healthy=%s, model_loaded=%s)",
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
        description="openpi WebSocket ↔ π0.5 gRPC bridge"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="gRPC endpoint (e.g. NLB_HOST:50051 or grpc://NLB_HOST:50051)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket listen port (default: 8000 — openpi_server default)",
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
