import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from filelock import FileLock
from pypdf import PdfReader
from tokenizers import Tokenizer

from ..config import manifest, settings
from ..integrations.telemetry import span
from ..logging import log

ort.disable_telemetry_events()

MODEL_SPECS = json.loads((Path(__file__).parent / "embedders.json").read_text())
INDEX_VERSION = "page-110-overlap-28-v4-preserve-lines"
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
            with (
                urllib.request.urlopen(url, timeout=60) as source,
                part.open("wb") as sink,
            ):
                shutil.copyfileobj(source, sink)
            if digest(part) != checksum:
                raise ValueError(f"Контрольная сумма файла {target.name} отличается от ожидаемой.")
            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)


class Encoder:
    def __init__(self, cache: Path, model: str):
        self.model, self.spec = model, MODEL_SPECS[model]
        for name, file in self.spec["files"].items():
            url = f"https://huggingface.co/{self.spec['repo']}/resolve/{self.spec['revision']}/{file['path']}"
            verified_download(url, cache / name, file["sha256"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(cache / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(cache / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.spec["max_length"])
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, texts: list[str], role: str) -> np.ndarray:
        with span(
            "rag.embed",
            "embedding",
            model=self.model,
            metadata={"role": role, "count": len(texts)},
        ) as obs:
            result = self._encode(texts, role)
            obs.update(metadata={"dimensions": result.shape[1]})
            return result

    def _encode(self, texts: list[str], role: str) -> np.ndarray:
        prefix = self.spec[role + "_prefix"]
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
        text = pdf_page.extract_text() or ""
        words = list(re.finditer(r"\S+", text))
        if len(words) < 12:
            continue
        for start in range(0, len(words), 82):
            part = words[start : start + 110]
            if len(part) < 12:
                break
            yield {
                "report": report["id"],
                "chunk_id": f"{report['id']}:p{page}:w{start}",
                "start_word": start,
                "title": report["title"],
                "date": report["date"],
                "page": page,
                "url": report["url"] + f"#page={page}",
                "local_url": f"/reports/{report['file']}#page={page}",
                "kind": "report",
                "text": text[part[0].start() : part[-1].end()],
            }
            if start + 110 >= len(words):
                break


class ReportIndex:
    def __init__(self, model: str | None = None):
        cfg = settings()
        self.model = model or cfg.embedding
        spec = MODEL_SPECS[self.model]
        reports = manifest()
        key = hashlib.sha256(
            (
                INDEX_VERSION
                + json.dumps(spec, sort_keys=True)
                + json.dumps(reports, sort_keys=True)
            ).encode()
        )
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
        self.fingerprint = key.hexdigest()
        self.encoder = Encoder(cfg.cache / "models" / self.model, self.model)
        folder = cfg.cache / ("index-" + key.hexdigest()[:16])
        with FileLock(str(folder) + ".lock"):
            if not folder.is_dir():
                log("rag.index_build_started", embedding=self.model)
                chunks = [row for meta, p in zip(reports, paths) for row in pdf_chunks(meta, p)]
                if not chunks:
                    raise ValueError("PDF не содержат извлекаемого текста.")
                embeddings = self.encoder.encode([c["text"] for c in chunks], "passage")
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
        log("rag.index_ready", embedding=self.model, fingerprint=self.fingerprint)
        self.index = faiss.read_index(str(folder / "index.faiss"))
        self.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        self.terms = [Counter(re.findall(r"\w+", c["text"].casefold())) for c in self.chunks]
        self.document_frequency = Counter(t for row in self.terms for t in row)
        self.average_length = sum(sum(row.values()) for row in self.terms) / len(self.terms)

    def search(self, question: str, k: int, mode: str | None = None) -> list[dict]:
        mode = mode or settings().rag_search
        if mode not in {"dense", "hybrid"}:
            raise ValueError("RAG_SEARCH_MODE must be dense or hybrid")
        vec = self.encoder.encode([question], "query")
        pool = min(k if mode == "dense" else max(40, k * 4), self.index.ntotal)
        sims, indices = self.index.search(vec, pool)
        if mode == "hybrid":
            tokens = set(re.findall(r"\w+", question.casefold()))
            scores = []
            n = len(self.terms)
            for i, row in enumerate(self.terms):
                score = 0.0
                length = sum(row.values())
                for term in tokens:
                    tf = row.get(term, 0)
                    df = self.document_frequency[term]
                    if tf:
                        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
                        score += (
                            idf
                            * tf
                            * 2.5
                            / (tf + 1.5 * (0.25 + 0.75 * length / self.average_length))
                        )
                if score > 0:
                    scores.append((i, score))
            lexical = [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:pool]]
            fused = {}
            for ranking in [indices[0].tolist(), lexical]:
                for rank, i in enumerate(ranking, 1):
                    if i >= 0:
                        fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
            best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
            return [
                dict(self.chunks[i], id=f"R{rank}", score=round(fused[i], 6), search_mode=mode)
                for rank, i in enumerate(best, 1)
            ]
        return [
            dict(self.chunks[int(i)], id=f"R{rank}", score=round(float(s), 4))
            for rank, (s, i) in enumerate(zip(sims[0], indices[0]), 1)
            if i >= 0
        ]


@lru_cache
def _index(model: str) -> ReportIndex:
    return ReportIndex(model)


def index(model: str | None = None) -> ReportIndex:
    with _init_lock:
        return _index(model or settings().embedding)
