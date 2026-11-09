import csv
import math
from datetime import date
from pathlib import Path

import numpy as np

from ..config import settings

SERIES = {"Brent": "MCOILBRENTEU", "WTI": "MCOILWTICO"}


def month_number(month: str) -> int:
    d = date.fromisoformat(month + "-01")
    return d.year * 12 + d.month - 1


def month_after(month: str, steps: int) -> str:
    year, m = divmod(month_number(month) + steps, 12)
    return f"{year:04d}-{m + 1:02d}"


def load_history(
    path: Path, benchmark: str, cutoff: str | None = None
) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        pairs = [
            (r["month"], float(r[benchmark]))
            for r in csv.DictReader(handle)
            if not cutoff or r["month"] <= cutoff
        ]
    if len(pairs) < 72:
        raise ValueError("Для калибровки нужно не менее 72 полных последовательных месяцев.")
    months, values = zip(*pairs)
    if any(month_after(a, 1) != b for a, b in zip(months, months[1:])):
        raise ValueError("В истории есть пропуск, повтор или нарушен порядок месяцев.")
    if months[-1] >= date.today().isoformat()[:7]:
        raise ValueError("В ряду обнаружен незавершённый или будущий месяц.")
    prices = np.asarray(values, dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Ожидаются положительные конечные среднемесячные цены.")
    return list(months), prices


def models(prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    # ARIMA(1,1,0) in logs: conditional least squares on differences, no drift.
    logs = np.log(prices[-120:])
    changes = np.diff(logs)
    x, y = changes[:-1], changes[1:]
    denominator = float(x @ x)
    phi = float(np.clip((x @ y) / denominator, -0.98, 0.98)) if denominator else 0.0
    arima = np.exp(logs[-1] + np.cumsum(changes[-1] * phi ** np.arange(1, horizon + 1)))
    # SES in logs: tune alpha on one-step errors within this training window only.
    alphas = np.linspace(0.05, 1.0, 20)
    levels = np.full_like(alphas, logs[0])
    losses = np.zeros_like(alphas)
    for obs in logs[1:]:
        losses += (obs - levels) ** 2
        levels += alphas * (obs - levels)
    ses = np.full(horizon, np.exp(levels[np.argmin(losses)]))
    return arima, ses


def forecast(
    benchmark: str,
    months: int,
    discount: float | None = None,
    cut: float | None = None,
    sensitivity: float | None = None,
    cutoff: str | None = None,
) -> dict:
    if not 1 <= months <= 12:
        raise ValueError("Поддерживается горизонт от 1 до 12 месяцев.")
    if benchmark not in {*SERIES, "Urals"}:
        raise ValueError("Выберите Brent или WTI. Для сценария Urals укажите дисконт к Brent.")
    if benchmark == "Urals" and discount is None:
        raise ValueError(
            "Для Urals укажите предполагаемый дисконт к Brent в USD/барр. Например, «Urals на 3 месяца при дисконте 12 долларов к Brent»."
        )
    if discount is not None and (
        benchmark != "Urals" or not math.isfinite(discount) or discount < 0
    ):
        raise ValueError("Положительный дисконт к Brent применим только к сценарию Urals.")
    if cut is not None and sensitivity is None:
        raise ValueError(
            "Укажите предполагаемую чувствительность цены в USD/барр. на сокращение на 1 млн б/с. Например, «Brent на 3 месяца при сокращении на 1 млн б/с и чувствительности 3 доллара на 1 млн б/с»."
        )
    if (sensitivity is not None and cut is None) or any(
        not math.isfinite(x) or x < 0 for x in (cut, sensitivity) if x is not None
    ):
        raise ValueError("Сценарий требует неотрицательных значений сокращения и чувствительности.")
    base = "Brent" if benchmark == "Urals" else benchmark
    cfg = settings()
    dates, prices = load_history(cfg.data / "prices/monthly.csv", base, cutoff)
    ar, ses = models(prices, months)
    point = (ar + ses) / 2
    # Each origin precedes its first target. The last target is already observed.
    origins = range(
        max(36, len(prices) - months - cfg.forecast_windows + 1),
        len(prices) - months + 1,
    )
    errors, naive_errors, log_errors = [], [], []
    for origin in origins:
        a, b = models(prices[:origin], months)
        pred = (a + b) / 2
        actual = prices[origin : origin + months]
        errors.append(np.abs(actual - pred))
        naive_errors.append(np.abs(actual - prices[origin - 1]))
        log_errors.append(np.abs(np.log(actual / pred)))
    n = len(log_errors)
    rank = min(n, math.ceil(0.9 * (n + 1))) - 1
    radius = np.sort(log_errors, axis=0)[rank]
    mae = np.mean(errors, axis=0)
    naive = np.mean(naive_errors, axis=0)
    shift = (cut or 0) * (sensitivity or 0) - (discount or 0)
    lower, upper = point * np.exp(-radius) + shift, point * np.exp(radius) + shift
    if min(lower) <= 0:
        raise ValueError(
            "Такой дисконт выводит диапазон за границы модели положительных цен. Уточните сценарий."
        )
    rows = []
    for h in range(months):
        rows.append(
            {
                "month": month_after(dates[-1], h + 1),
                "arima": round(ar[h] + shift, 2),
                "ses": round(ses[h] + shift, 2),
                "point": round(point[h] + shift, 2),
                "lower": round(lower[h], 2),
                "upper": round(upper[h], 2),
                "mae": round(mae[h], 2),
                "naive_mae": round(naive[h], 2),
            }
        )
    return {
        "benchmark": benchmark,
        "base": base,
        "as_of": dates[-1],
        "last": float(prices[-1]),
        "unit": "USD/барр.",
        "level": 0.9,
        "origins": n,
        "rows": rows,
        "discount": discount,
        "cut": cut,
        "sensitivity": sensitivity,
        "source_url": "https://fred.stlouisfed.org/series/" + SERIES[base],
    }
