"""Bound upload bodies before Starlette can spool multipart files to disk."""

from io import BytesIO

from starlette.responses import JSONResponse

from .config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MIB


class UploadLimitMiddleware:
    def __init__(self, app, max_bytes=MAX_UPLOAD_BYTES + 1024 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/knowledge/documents"
        ):
            return await self.app(scope, receive, send)

        async def reject():
            await JSONResponse(
                {"detail": f"Запрос слишком большой. Максимальный файл — {MAX_UPLOAD_MIB} МБ."},
                status_code=413,
            )(scope, receive, send)

        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > self.max_bytes:
                return await reject()
        except ValueError:
            return await JSONResponse({"detail": "Некорректная длина запроса."}, 400)(
                scope, receive, send
            )
        body_buffer, size = BytesIO(), 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            size += len(data)
            if size > self.max_bytes:
                return await reject()
            body_buffer.write(data)
            if not message.get("more_body", False):
                break
        # Replay in chunks so the multipart parser can spool large files to disk.
        body_buffer.seek(0)

        async def bounded_receive():
            if body_buffer.closed:
                return await receive()
            chunk = body_buffer.read(64 * 1024)
            more_body = body_buffer.tell() < size
            if not more_body:
                body_buffer.close()
            return {"type": "http.request", "body": chunk, "more_body": more_body}

        await self.app(scope, bounded_receive, send)
