#тест через терминал

import argparse
import csv
import io
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import httpx
from filelock import FileLock

from .config import settings
from .forecast import SERIES, load_history
from .rag import index
from .web import FRED_CSV


def refresh_prices():
    cfg = settings()
    target = cfg.data / "prices/monthly.csv"
    with FileLock(str(target) + ".lock"):
        r = httpx.get(
            FRED_CSV,
            params={"id": ",".join(SERIES.values())},
            timeout=cfg.timeout,
            follow_redirects=True,
        )
        r.raise_for_status()
        cutoff = date.today().replace(day=1).isoformat()
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".csv", mode="w", newline="", encoding="utf-8", delete=False
        ) as temp:
            part = Path(temp.name)
        try:
            with part.open("w", encoding="utf-8", newline="") as sink:
                writer = csv.writer(sink, lineterminator="\n")
                writer.writerow(["month", *SERIES])
                for row in csv.DictReader(io.StringIO(r.text)):
                    dt = row["observation_date"]
                    if "2010-01-01" <= dt < cutoff:
                        writer.writerow([dt[:7], *(row[s] for s in SERIES.values())])
            for grade in SERIES:
                dates, _ = load_history(part, grade)
                if target.exists() and dates[-1] < load_history(target, grade)[0][-1]:
                    raise ValueError("Полученный ряд короче сохранённого. Обновление отменено.")
            os.replace(part, target)
            print(f"{len(dates)} месяцев; последнее наблюдение {dates[-1]}.")
        finally:
            part.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="PetroDesk")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="Проверить PDF, загрузить E5, собрать FAISS")
    sub.add_parser("refresh-prices", help="Обновить месячные цены из EIA/FRED")
    q = sub.add_parser("ask", help="Задать вопрос графу агента")
    q.add_argument("question")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(f"Индекс готов: {index().index.ntotal} фрагментов.")
    elif args.cmd == "refresh-prices":
        refresh_prices()
    else:
        from .agent import ask

        print(json.dumps(ask(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
