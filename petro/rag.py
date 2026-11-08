import hashlib
import json
import os
import shutil
import tempfile
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from filelock import FileLock
from pypdf import PdfReader
from tokenizers import Tokenizer

from .config import manifest, settings

MODEL_REV = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
MODEL_URL = f"https://huggingface.co/Xenova/multilingual-e5-small/resolve/{MODEL_REV}/"
MODEL_FILES = {
    "tokenizer.json": (
        "tokenizer.json",
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    "model.onnx": (
        "onnx/model_quantized.onnx",
        "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
    ),
}
INDEX_VERSION = "page-110-overlap-28-v2"
_init_lock = threading.Lock()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_download(url: str, target: Path, checksum: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if target.exists() and digest(target) == checksum:
            return
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            part = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as source, part.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            if digest(part) != checksum:
                raise ValueError(f"Контрольная сумма файла {target.name} отличается от ожидаемой.")
            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)


class Encoder:
    def __init__(self, cache: Path):
        for name, (relative, checksum) in MODEL_FILES.items():
            verified_download(MODEL_URL + relative, cache / name, checksum)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(cache / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(cache / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        output = []
        for start in range(0, len(texts), 12):
            items = self.tokenizer.encode_batch([prefix + t for t in texts[start : start + 12]])
            ids = np.array([e.ids for e in items], dtype="int64")
            mask = np.array([e.attention_mask for e in items], dtype="int64")
            feeds = {"input_ids": ids, "attention_mask": mask}
            if any(inp.name == "token_type_ids" for inp in self.session.get_inputs()):
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            pooled = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
            output.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.ascontiguousarray(np.vstack(output), dtype="float32")


def pdf_chunks(report: dict, path: Path):
    for page, pdf_page in enumerate(PdfReader(path).pages, start=1):
        words = (pdf_page.extract_text() or "").split()
        if len(words) < 12:
            continue
        for start in range(0, len(words), 82):
            part = words[start : start + 110]
            if len(part) < 12:
                break
            yield {
                "report": report["id"],
                "title": report["title"],
                "date": report["date"],
                "page": page,
                "url": report["url"] + f"#page={page}",
                "local_url": f"/reports/{report['file']}#page={page}",
                "kind": "report",
                "text": " ".join(part),
            }
            if start + 110 >= len(words):
                break


class ReportIndex:
    def __init__(self):
        cfg = settings()
        reports = manifest()
        key = hashlib.sha256((INDEX_VERSION + MODEL_REV + json.dumps(reports, sort_keys=True)).encode())
        paths = []
        for item in reports:
            path = cfg.data / "reports" / item["file"]
            actual = digest(path)
            if item.get("sha256") and actual != item["sha256"]:
                raise ValueError(
                    f"Изменился зарегистрированный PDF {path.name}. Проверьте источник и reports.json."
                )
            key.update(actual.encode())
            paths.append(path)
        cfg.cache.mkdir(parents=True, exist_ok=True)
        self.encoder = Encoder(cfg.cache / "model")
        folder = cfg.cache / ("index-" + key.hexdigest()[:16])
        with FileLock(str(folder) + ".lock"):
            if not folder.is_dir():
                chunks = [row for meta, p in zip(reports, paths) for row in pdf_chunks(meta, p)]
                if not chunks:
                    raise ValueError("PDF не содержат извлекаемого текста.")
                embeddings = self.encoder.encode([c["text"] for c in chunks], "passage: ")
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                tmp = Path(tempfile.mkdtemp(dir=cfg.cache))
                try:
                    faiss.write_index(index, str(tmp / "index.faiss"))
                    (tmp / "chunks.json").write_text(
                        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
                    )
                    tmp.rename(folder)
                finally:
                    if tmp.exists():
                        shutil.rmtree(tmp)
        self.index = faiss.read_index(str(folder / "index.faiss"))
        self.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))

    def search(self, question: str, k: int) -> list[dict]:
        vec = self.encoder.encode([question], "query: ")
        sims, indices = self.index.search(vec, min(k, self.index.ntotal))
        return [
            dict(self.chunks[int(i)], id=f"R{rank}", score=round(float(s), 4))
            for rank, (s, i) in enumerate(zip(sims[0], indices[0]), 1)
            if i >= 0
        ]


@lru_cache
def _index() -> ReportIndex:
    return ReportIndex()


def index() -> ReportIndex:
    with _init_lock:
        return _index()
