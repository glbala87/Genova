"""WebSocket streaming API for Genova model inference.

Provides real-time streaming endpoints for token-by-token sequence generation
and progress-tracking during long batch predictions.

Usage::

    from genova.api.websocket import WebSocketManager, register_websocket_routes

    ws_manager = WebSocketManager()
    register_websocket_routes(app, ws_manager)

Clients connect via ``/ws/generate`` or ``/ws/predict`` and exchange JSON
messages following the protocol::

    # Client -> Server
    {"type": "generate", "data": {"sequence": "ACGT", "max_length": 100, ...}}
    {"type": "predict",  "data": {"sequences": ["ACGT"], "task": "variant", ...}}
    {"type": "ping"}

    # Server -> Client
    {"type": "token",    "data": {"token": "A", "position": 42, "prob": 0.93}}
    {"type": "progress", "data": {"current": 5, "total": 100, "pct": 5.0}}
    {"type": "result",   "data": {...}}
    {"type": "error",    "data": {"message": "...", "code": "..."}}
    {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from genova.api.inference import InferenceEngine


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _msg(msg_type: str, data: Any = None) -> str:
    """Serialise a protocol message."""
    return json.dumps({"type": msg_type, "data": data})


def _error_msg(message: str, code: str = "INTERNAL_ERROR") -> str:
    return _msg("error", {"message": message, "code": code})


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------

@dataclass
class _Connection:
    """Tracks a single WebSocket client."""

    ws: Any  # WebSocket
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    connected_at: float = field(default_factory=time.monotonic)
    last_pong: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# WebSocketManager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Manage WebSocket connections with ping/pong keep-alive.

    Parameters
    ----------
    ping_interval : float
        Seconds between server-initiated ping frames (default 30).
    pong_timeout : float
        Seconds to wait for a pong before considering the client dead (default 10).
    max_connections : int
        Maximum concurrent WebSocket connections (default 100).
    """

    def __init__(
        self,
        ping_interval: float = 30.0,
        pong_timeout: float = 10.0,
        max_connections: int = 100,
    ) -> None:
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout
        self.max_connections = max_connections
        self._connections: Dict[str, _Connection] = {}
        self._ping_task: Optional[asyncio.Task] = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self, ws: Any) -> _Connection:
        """Accept a WebSocket and register it.

        Raises
        ------
        RuntimeError
            If the maximum connection limit has been reached.
        """
        if len(self._connections) >= self.max_connections:
            await ws.close(code=1013, reason="Max connections reached")
            raise RuntimeError("Max WebSocket connections reached")
        await ws.accept()
        conn = _Connection(ws=ws)
        self._connections[conn.id] = conn
        logger.info("WebSocket connected: {}", conn.id)
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.ensure_future(self._ping_loop())
        return conn

    async def disconnect(self, conn: _Connection) -> None:
        """Remove a connection from the manager."""
        self._connections.pop(conn.id, None)
        logger.info("WebSocket disconnected: {}", conn.id)

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    # -- broadcasting --------------------------------------------------------

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        dead: List[str] = []
        for cid, conn in self._connections.items():
            try:
                await conn.ws.send_text(message)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self._connections.pop(cid, None)

    # -- ping / pong ---------------------------------------------------------

    async def _ping_loop(self) -> None:
        """Periodically send pings and prune dead connections."""
        while self._connections:
            await asyncio.sleep(self.ping_interval)
            dead: List[str] = []
            now = time.monotonic()
            for cid, conn in list(self._connections.items()):
                # Check pong timeout
                if now - conn.last_pong > self.ping_interval + self.pong_timeout:
                    dead.append(cid)
                    continue
                try:
                    await conn.ws.send_text(_msg("ping"))
                except Exception:
                    dead.append(cid)
            for cid in dead:
                c = self._connections.pop(cid, None)
                if c:
                    try:
                        await c.ws.close(code=1001, reason="Ping timeout")
                    except Exception:
                        pass
                    logger.info("Pruned dead connection: {}", cid)

    def _handle_pong(self, conn: _Connection) -> None:
        conn.last_pong = time.monotonic()

    # -- streaming helpers ---------------------------------------------------

    async def stream_generate(
        self,
        conn: _Connection,
        engine: InferenceEngine,
        sequence: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> None:
        """Stream token-by-token sequence generation.

        Sends ``token`` messages for each generated nucleotide and a final
        ``result`` message with the complete sequence.
        """
        import torch
        import numpy as np

        engine._ensure_loaded()
        assert engine.tokenizer is not None and engine.model is not None

        generated_tokens: List[str] = list(sequence)

        # Tokenise the seed
        encoded = engine.tokenizer.encode(sequence)
        input_ids = torch.tensor(
            [encoded], dtype=torch.long, device=engine.device
        )

        for step in range(max_length):
            with torch.no_grad():
                outputs = engine.model(input_ids=input_ids)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("last_hidden_state"))
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = getattr(outputs, "logits", outputs)

                # Take last position logits
                next_logits = logits[0, -1, :] / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0 and top_k < next_logits.size(0):
                    values, _ = torch.topk(next_logits, top_k)
                    min_val = values[-1]
                    next_logits[next_logits < min_val] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                prob = float(probs[next_id].item())

                # Decode token
                token_str = engine.tokenizer.decode([next_id.item()])

                generated_tokens.append(token_str)

                await conn.ws.send_text(_msg("token", {
                    "token": token_str,
                    "position": len(generated_tokens) - 1,
                    "prob": round(prob, 6),
                    "step": step + 1,
                    "total_steps": max_length,
                }))

                # Append for next iteration
                input_ids = torch.cat(
                    [input_ids, next_id.unsqueeze(0)], dim=1
                )

        full_sequence = "".join(generated_tokens)
        await conn.ws.send_text(_msg("result", {
            "sequence": full_sequence,
            "length": len(full_sequence),
            "num_generated": max_length,
        }))

    async def stream_predict(
        self,
        conn: _Connection,
        engine: InferenceEngine,
        sequences: List[str],
        task: str = "embed",
        batch_size: int = 8,
    ) -> None:
        """Stream progress updates during batch prediction.

        Sends ``progress`` messages as batches complete and a final ``result``
        message with all predictions.
        """
        total = len(sequences)
        all_results: List[Any] = []

        for start in range(0, total, batch_size):
            batch = sequences[start: start + batch_size]

            if task == "variant":
                # Expect pairs: even indices = ref, odd = alt
                mid = len(batch) // 2
                refs = batch[:mid] if mid > 0 else batch
                alts = batch[mid:] if mid > 0 else batch
                preds = engine.predict_variant(refs, alts)
                all_results.extend(preds)
            elif task == "expression":
                preds = engine.predict_expression(batch)
                all_results.extend([p.tolist() for p in preds])
            elif task == "methylation":
                preds = engine.predict_methylation(batch)
                all_results.extend([p.tolist() for p in preds])
            else:
                preds = engine.embed(batch)
                all_results.extend([p.tolist() for p in preds])

            done = min(start + batch_size, total)
            pct = round(done / total * 100, 1)
            await conn.ws.send_text(_msg("progress", {
                "current": done,
                "total": total,
                "pct": pct,
                "task": task,
            }))

            # Yield control so other coroutines can run
            await asyncio.sleep(0)

        await conn.ws.send_text(_msg("result", {
            "predictions": all_results,
            "num_sequences": total,
            "task": task,
        }))


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_websocket_routes(
    app: "FastAPI",
    manager: Optional[WebSocketManager] = None,
) -> WebSocketManager:
    """Register ``/ws/generate`` and ``/ws/predict`` WebSocket routes.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    manager : WebSocketManager, optional
        An existing manager. A new one is created if *None*.

    Returns
    -------
    WebSocketManager
        The active manager instance.
    """
    if not _HAS_FASTAPI:
        logger.warning("FastAPI not installed; WebSocket routes not registered.")
        return manager or WebSocketManager()

    if manager is None:
        manager = WebSocketManager()

    @app.websocket("/ws/generate")
    async def ws_generate(ws: WebSocket) -> None:
        """WebSocket endpoint for streaming sequence generation."""
        conn = await manager.connect(ws)
        engine: Optional[InferenceEngine] = getattr(
            ws.app.state, "engine", None
        )
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(_error_msg("Invalid JSON", "PARSE_ERROR"))
                    continue

                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "ping":
                    manager._handle_pong(conn)
                    await ws.send_text(_msg("pong"))
                elif msg_type == "generate":
                    if engine is None or not engine.is_loaded():
                        await ws.send_text(
                            _error_msg("Model not loaded", "MODEL_UNAVAILABLE")
                        )
                        continue
                    await manager.stream_generate(
                        conn,
                        engine,
                        sequence=data.get("sequence", "ACGT"),
                        max_length=data.get("max_length", 100),
                        temperature=data.get("temperature", 1.0),
                        top_k=data.get("top_k", 50),
                    )
                else:
                    await ws.send_text(
                        _error_msg(f"Unknown message type: {msg_type}", "UNKNOWN_TYPE")
                    )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket generate error: {}", exc)
            try:
                await ws.send_text(_error_msg(str(exc)))
            except Exception:
                pass
        finally:
            await manager.disconnect(conn)

    @app.websocket("/ws/predict")
    async def ws_predict(ws: WebSocket) -> None:
        """WebSocket endpoint for streaming batch predictions."""
        conn = await manager.connect(ws)
        engine: Optional[InferenceEngine] = getattr(
            ws.app.state, "engine", None
        )
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(_error_msg("Invalid JSON", "PARSE_ERROR"))
                    continue

                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "ping":
                    manager._handle_pong(conn)
                    await ws.send_text(_msg("pong"))
                elif msg_type == "predict":
                    if engine is None or not engine.is_loaded():
                        await ws.send_text(
                            _error_msg("Model not loaded", "MODEL_UNAVAILABLE")
                        )
                        continue
                    await manager.stream_predict(
                        conn,
                        engine,
                        sequences=data.get("sequences", []),
                        task=data.get("task", "embed"),
                        batch_size=data.get("batch_size", 8),
                    )
                else:
                    await ws.send_text(
                        _error_msg(f"Unknown message type: {msg_type}", "UNKNOWN_TYPE")
                    )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket predict error: {}", exc)
            try:
                await ws.send_text(_error_msg(str(exc)))
            except Exception:
                pass
        finally:
            await manager.disconnect(conn)

    logger.info("WebSocket routes registered: /ws/generate, /ws/predict")
    return manager
