"""Pytest configuration ensuring offline asyncio unit tests run cleanly even in restricted Windows sandboxes."""

from __future__ import annotations

import asyncio
import selectors
import sys
import threading
import time


def _loopback_socketpair_healthy() -> bool:
    if sys.platform != "win32":
        return True
    result = {"ok": False}

    def _probe() -> None:
        import socket

        try:
            a, b = socket.socketpair()
            a.close()
            b.close()
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=0.25)
    return bool(result["ok"])


if sys.platform == "win32" and not _loopback_socketpair_healthy():

    class _SafeSelectSelector(selectors.SelectSelector):
        def _select(self, r, w, x, timeout=None):  # type: ignore[override]
            if not r and not w:
                if timeout and timeout > 0:
                    time.sleep(min(float(timeout), 0.05))
                return [], [], []
            return super()._select(r, w, x, timeout)

    class _NoPipeSelectorLoop(asyncio.SelectorEventLoop):
        def _make_self_pipe(self) -> None:
            self._ssock = None  # type: ignore[assignment]
            self._csock = None  # type: ignore[assignment]

        def _close_self_pipe(self) -> None:
            pass

        def _write_to_self(self) -> None:
            pass

    class _SandboxEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self) -> asyncio.AbstractEventLoop:
            return _NoPipeSelectorLoop(_SafeSelectSelector())

    asyncio.set_event_loop_policy(_SandboxEventLoopPolicy())
