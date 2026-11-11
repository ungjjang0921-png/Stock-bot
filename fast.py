"""
⚡ 빠른 갱신 (10분마다, 1분 안에 끝남) — 사이트의 '지금 급등 중'을 거의 실시간으로
════════════════════════════════════════════
TradingView 스캐너에서 프장·정규장·애프터 급등 소형주를 받아 docs/live.json 으로 저장.
무거운 분석(watchlist.py)은 30분마다, 이건 급등 목록만 빠르게.
"""

import json
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
OUT = Path(__file__).parent / "docs" / "live.json"
MIN_PRICE, MAX_PRICE = 0.01, 20.0      # 1센트 동전주까지
MAX_MCAP = 500e6
COLS = ["description", "close", "change", "volume", "market_cap_basic", "float_shares_outstanding",
        "premarket_close", "premarket_change", "premarket_volume",
        "postmarket_close", "postmarket_change", "postmarket_volume"]
HI = {"pre": "premarket_high", "post": "postmarket_high", "reg": "high"}


def mode(now):
    t = now.time()
    if dtime(4, 0) <= t < dtime(9, 30):
        return "pre", "premarket_change", "premarket_volume", "premarket_close"
    if dtime(9, 30) <= t < dtime(16, 0):
        return "reg", "change", "volume", "close"
    return "post", "postmarket_change", "postmarket_volume", "postmarket_close"


def scan(chg_col, vol_col, hi_col, n=60):
    for cols in (COLS + [hi_col], COLS):          # 고가 컬럼이 안 되면 빼고 다시
        rows = _scan(cols, chg_col, vol_col, n)
        if rows is not None:
            return cols, rows
    return COLS, []


def _scan(cols, chg_col, vol_col, n):
    body = {"markets": ["america"], "columns": cols, "range": [0, n],
            "filter": [{"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                       {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                       {"left": "close", "operation": "in_range", "right": [MIN_PRICE, MAX_PRICE]},
                       {"left": vol_col, "operation": "greater", "right": 30_000},
                       {"left": chg_col, "operation": "greater", "right": 5}],
            "sort": {"sortBy": chg_col, "sortOrder": "desc"}}
    for _ in range(3):
        try:
            r = requests.post("https://scanner.tradingview.com/america/scan", json=body, timeout=25,
                              headers={"User-Agent": "Mozilla/5.0"})
            j = r.json()
            if "data" in j:
                return j["data"]
            print("스캔 오류:", str(j)[:150])
            return None
        except Exception as e:
            print("스캔 실패:", e)
            time.sleep(3)
    return None


def main():
    now = datetime.now(ET)
    m, cc, vc, pc = mode(now)
    rows = []
    cols, data = scan(cc, vc, HI[m])
    for row in data:
        d = dict(zip(cols, row["d"]))
        mc = d.get("market_cap_basic")
        if mc and mc > MAX_MCAP:
            continue
        px = d.get(pc) or d.get("close")
        if not px or d.get(cc) is None:
            continue
        rows.append([row["s"].split(":")[-1].replace(".", "-"), (d.get("description") or "")[:40],
                     round(px, 4), round(d[cc], 1), int(d.get(vc) or 0),
                     round(d["float_shares_outstanding"]) if d.get("float_shares_outstanding") else None,
                     round(d[HI[m]], 4) if d.get(HI[m]) else None])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"ts": int(time.time()), "mode": m, "et": now.strftime("%m/%d %H:%M"), "rows": rows[:40]},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{m} {len(rows)}종목 → {OUT}")


if __name__ == "__main__":
    main()
