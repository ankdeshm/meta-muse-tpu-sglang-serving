#!/usr/bin/env python3
"""Lightweight async least-inflight streaming HTTP gateway for multi-replica SGLang servers on one TPU host."""

from __future__ import annotations

import argparse
import asyncio
import itertools
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


class LeastInflightProxy:
    def __init__(self, backends: list[str]) -> None:
        self.backends = [b.rstrip("/") for b in backends]
        self.inflight = [0] * len(self.backends)
        self._rr = itertools.count()
        self.session: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        conn = TCPConnector(limit=0, force_close=False, enable_cleanup_closed=True)
        self.session = ClientSession(connector=conn, timeout=ClientTimeout(total=7200))

    async def stop(self, _app: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    def pick_backend(self) -> int:
        # Pick minimum inflight; break ties round-robin so conc=1 alternates evenly or hits idle replica
        n = len(self.backends)
        offset = next(self._rr) % n
        best_idx = offset
        best_val = self.inflight[offset]
        for k in range(1, n):
            idx = (offset + k) % n
            if self.inflight[idx] < best_val:
                best_val = self.inflight[idx]
                best_idx = idx
        return best_idx

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        path_qs = request.rel_url.path_qs
        # Broadcast cache flush to all replicas
        if request.method == "POST" and request.path in ("/flush_cache",):
            body = await request.read()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
            resps = await asyncio.gather(
                *[self.session.post(f"{b}{path_qs}", data=body, headers=headers) for b in self.backends],
                return_exceptions=True,
            )
            for r in resps:
                if not isinstance(r, Exception):
                    await r.release()
            return web.Response(status=200, text="flushed")

        idx = self.pick_backend()
        self.inflight[idx] += 1
        target_url = f"{self.backends[idx]}{path_qs}"
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        try:
            async with self.session.request(request.method, target_url, data=body, headers=headers) as upstream:
                out_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding")
                }
                stream = web.StreamResponse(status=upstream.status, headers=out_headers)
                await stream.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if chunk:
                        await stream.write(chunk)
                await stream.write_eof()
                return stream
        finally:
            self.inflight[idx] -= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29999)
    ap.add_argument("--backends", required=True, help="Comma-separated backend base URLs")
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    proxy = LeastInflightProxy(backends)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
