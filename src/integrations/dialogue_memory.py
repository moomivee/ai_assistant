"""Short-lived, request-scoped forecast clarifications in the Ouroboros dialogue store."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import secrets
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import time

from ouroboros import dialogue_store, store

from ..config import settings

logger = logging.getLogger(__name__)

TTL_SECONDS = 24 * 60 * 60
_TOKEN = re.compile(r"[0-9a-f]{64}")
_EXPIRED = "Уточнение завершено или недоступно в этом режиме. Начните новый вопрос."


class DialogueConflict(ValueError):
    pass


class DialogueBusy(DialogueConflict):
    pass


class DialogueStorageError(RuntimeError):
    pass


@contextmanager
def _storage_errors():
    try:
        yield
    except (OSError, store.StoreError) as exc:
        raise DialogueStorageError("Хранилище уточнений временно недоступно.") from exc


def _root() -> Path:
    return settings().data / "ouroboros" / "clarifications"


def _bucket_name(day: int) -> str:
    return f"day-{day:08x}"


@contextmanager
def _directory_lease(bucket: Path, mode: int):
    fd = os.open(bucket, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _prune_expired(root: Path, day: int) -> None:
    # A session started just before midnight is valid until the following midnight.
    # Retire whole expired stores without rewriting an Ouroboros append-only history.
    for bucket in root.glob("day-*"):
        if not re.fullmatch(r"day-[0-9a-f]{8}", bucket.name) or bucket.is_symlink():
            continue
        if int(bucket.name[4:], 16) >= day - 1:
            continue
        try:
            with _directory_lease(bucket, fcntl.LOCK_EX):
                store.close_connections_under(bucket)
                shutil.rmtree(bucket)
        except (FileNotFoundError, BlockingIOError):
            continue  # Another worker already retired this bucket or still holds a turn.
        except (OSError, store.StoreError) as exc:
            logger.warning(
                "Не удалось очистить просроченный архив уточнений %s: %s", bucket.name, exc
            )


def _token(day: int, room: int) -> str:
    return f"{day:08x}{room:016x}{secrets.token_hex(20)}"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _write(bucket: Path, room: int, row: dict) -> None:
    with _storage_errors():
        dialogue_store.append(bucket, "chat", {"chat_id": room, **row}, emit_live=False)


def _save(bucket: Path, room: int, scope: dict, expires: int, snapshot: dict) -> str:
    token = _token(int(bucket.name[4:], 16), room)
    _write(
        bucket,
        room,
        {
            "type": "forecast_clarification",
            "token_hash": _digest(token),
            "expires": expires,
            "scope": scope,
            "snapshot": snapshot,
        },
    )
    return token


def open_dialogue(snapshot: dict, scope: dict) -> str:
    now = int(time())
    day = now // TTL_SECONDS
    root = _root()
    bucket = root / _bucket_name(day)
    with _storage_errors():
        bucket.mkdir(parents=True, mode=0o700, exist_ok=True)
        _prune_expired(root, day)
        with _directory_lease(bucket, fcntl.LOCK_SH):
            while True:
                room = secrets.randbits(63)
                try:
                    # Creation reserves a unique room. Unknown tokens never create lock files.
                    fd = os.open(bucket / f"{room:016x}.lock", os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                break
            return _save(bucket, room, scope, now + TTL_SECONDS, snapshot)


class DialogueTurn:
    def __init__(self, bucket: Path, room: int, row: dict):
        self.snapshot = row["snapshot"]
        self._bucket = bucket
        self._room = room
        self._scope = row["scope"]
        self._expires = row["expires"]
        self._active = True

    def advance(self, snapshot: dict) -> str:
        if not self._active or self._expires <= time():
            raise DialogueConflict(_EXPIRED)
        token = _save(self._bucket, self._room, self._scope, self._expires, snapshot)
        self._active = False
        return token

    def close(self) -> None:
        if not self._active:
            raise DialogueConflict(_EXPIRED)
        _write(self._bucket, self._room, {"type": "forecast_clarification_closed"})
        self._active = False


@contextmanager
def resume_dialogue(token: str, scope: dict):
    if not _TOKEN.fullmatch(token):
        raise DialogueConflict(_EXPIRED)
    day, room = int(token[:8], 16), int(token[8:24], 16)
    today = int(time()) // TTL_SECONDS
    if day not in (today, today - 1) or room >= 2**63:
        raise DialogueConflict(_EXPIRED)
    bucket = _root() / _bucket_name(day)
    with ExitStack() as locks:
        with _storage_errors():
            try:
                locks.enter_context(_directory_lease(bucket, fcntl.LOCK_SH))
                room_lock = locks.enter_context((bucket / f"{room:016x}.lock").open("rb"))
                fcntl.flock(room_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except FileNotFoundError:
                raise DialogueConflict(_EXPIRED) from None
            except BlockingIOError:
                raise DialogueBusy("Этот ответ на уточнение уже обрабатывается.") from None
            rows = dialogue_store.room_candidates(bucket, "chat", room, limit=1)
            row = rows[0].row if rows else {}
            if (
                not secrets.compare_digest(row.get("token_hash", ""), _digest(token))
                or row["expires"] <= time()
                or row["scope"] != scope
            ):
                raise DialogueConflict(_EXPIRED)
            turn = DialogueTurn(bucket, room, row)
        try:
            yield turn
        finally:
            # A model failure leaves an uncommitted token usable; the caller cannot write
            # outside this lease. A killed worker also releases both kernel locks.
            turn._active = False
