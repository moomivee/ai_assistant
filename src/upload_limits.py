"""Bound upload bodies before Starlette can spool multipart files to disk."""

from starlette.responses import JSONResponse


class UploadLimitMiddleware:
    def __init__(self, app, max_bytes=21 * 1024 * 1024):
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
                {"detail": "Запрос слишком большой. Максимальный файл — 20 МБ."}, status_code=413
            )(scope, receive, send)

        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > self.max_bytes:
                return await reject()
        except ValueError:
            return await JSONResponse({"detail": "Некорректная длина запроса."}, 400)(
                scope, receive, send
            )
        body_buffer, size = bytearray(), 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            size += len(data)
            if size > self.max_bytes:
                return await reject()
            body_buffer.extend(data)
            if not message.get("more_body", False):
                break
        body = bytes(body_buffer)
        sent = False

        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)
