"""Wall-clock limit for network I/O on the short quote fallback path."""

import asyncio
from contextlib import closing
from time import monotonic

import httpx


def request_before(deadline, method, url, **kwargs):
    """Cancel connection, redirects and body reads at one absolute deadline."""

    async def send():
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise httpx.ReadTimeout("Publication search deadline")
        try:
            async with asyncio.timeout(remaining), httpx.AsyncClient(timeout=remaining) as client:
                return await client.request(method, url, **kwargs)
        except TimeoutError as exc:
            raise httpx.ReadTimeout("Publication search deadline") from exc

    # The synchronous tool owns this loop. Closing it releases the caller even
    # when an OS DNS lookup in the default executor finishes after cancellation.
    with closing(asyncio.new_event_loop()) as loop:
        return loop.run_until_complete(send())
