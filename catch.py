"""
포착 엔진 (RabbitStock 방식) — 5개 엔진이 각자 순위를 내고, 여러 엔진이 같이 잡은 종목을 위로
════════════════════════════════════════════
👁 프리뷰  : 장중 흐름 — 비슷한 장중 상황에서 본장 +20%를 친 비율 + VWAP·거래량
📈 시그널  : 학습 모델 — 1·5·10거래일 급등 판별 점수(백분위)와 검증 확률, 차트 패턴
🏅 베스트  : 유사 사례 — 지금과 비슷했던 과거 날들 150건 중 +30% 간 비율, +30%가 -10% 손절보다 먼저 닿은 비율
🔥 버닝업  : 강하게 치솟는 종목 — 감지점수 = 5일 확률 순위 70 + 당일 상승 20 + 유통 작음 10
🩳 스퀴즈  : 숏스퀴즈 점수
정렬: 많이 포착된 순(엔진 수) → 평균 순위
기록: '오늘'·'1거래일' 포착은 하루 한 번 고정해서 실제 결과로 엔진별 적중률을 매김
"""

import json
import logging
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np

log = logging.getLogger("catch")

TOP = 20                       # 엔진별 상위 몇 개를 '포착'으로 볼지
HIT = {"0": 15, "1": 20}       # 적중 기준: 오늘(진입가 대비 당일 고가 +15%), 1거래일(종가 대비 다음날 고가 +20%)
ENG = ["preview", "signal", "best", "burn", "squeeze"]
ENG_KO = {"preview": "프리뷰", "signal": "시그널", "best": "베스트", "burn": "버닝업", "squeeze": "스퀴즈"}


def _risk(it: dict) -> list[str]:
    names = {t["name"] for t in it.get("tags", [])}
    return [n for n in ("부도 위험", "ATM·공모", "공모 준비", "물량 출회") if n in names]


def build(items: list, pred: dict, squeeze: list, now_et: datetime, live: bool) -> dict:
    by = {it["t"]: it for it in items}
    sc = (pred or {}).get("scores") or {}
    out = {"live": live, "hz": {}}
    for hz in ("0", "1", "5", "10"):
        H = "1" if hz == "0" else hz
        S = sc.get(H) or {}
        S5 = sc.get("5") or {}
        eng = {}
        # 👁 프리뷰 — 장중(오늘·1거래일만)
        if hz in ("0", "1"):
            rows = []
            for it in items:
                pb = it.get("prob") or {}
                pre_mode = now_et.time() < dtime(9, 30)
                pr = pb.get("pre") if pre_mode and pb.get("pre") is not None else pb.get("reg")
                sn = it.get("snap") or {}
                if pr is None or "부도 위험" in _risk(it):
                    continue
                s = pr
                if sn.get("vwap") and it["price"] >= sn["vwap"]:
                    s *= 1.15
                if (sn.get("relvol") or 0) >= 2:
                    s *= 1.1
                rows.append((s, it["t"], {"reg": round(pr, 1), "pre": pb.get("pre"),
                                          "vwap": bool(sn.get("vwap") and it["price"] >= sn["vwap"]),
                                          "rv": round(sn["relvol"], 1) if sn.get("relvol") else None}))
            eng["preview"] = rows
        # 📈 시그널 — 모델 백분위
        rows = []
        for t, r in S.items():
            if t in by and "부도 위험" in _risk(by[t]):
                continue
            rows.append((r["p"], t, {"p": r["p"], "pat": r["pat"],
                                     "probs": {k: (sc.get(k, {}).get(t) or {}).get("prob") for k in ("1", "5", "10")},
                                     "q": r["q"]}))
        eng["signal"] = rows
        # 🏅 베스트 — 유사 사례 선도달률 (모델 상위 30%만)
        rows = []
        for t, r in S.items():
            if r["p"] < 70 or (t in by and _risk(by[t])):
                continue
            rows.append((r["ft"] * 2 + r["r30"], t, {"ft": r["ft"], "r30": r["r30"], "q": r["q"], "cq": r.get("cq"),
                                                     "sim": r["sim"]}))
        eng["best"] = rows
        # 🔥 버닝업 — 오늘 +10% 이상 치솟는 종목
        rows = []
        for t, r in S.items():
            it = by.get(t)
            chg = ((it or {}).get("snap") or {}).get("chg")
            chg = chg if chg is not None else r["ret1"]
            if chg is None or chg < 10 or (it and "부도 위험" in _risk(it)):
                continue
            p5 = (S5.get(t) or {}).get("p", r["p"])
            fl = (it or {}).get("float") or 9e9
            fl_pt = 1 if fl < 5e6 else 0.6 if fl < 20e6 else 0.3 if fl < 50e6 else 0
            det = round(70 * p5 / 100 + 20 * min(chg / 50, 1) + 10 * fl_pt, 1)
            b = r.get("burn") or {}
            rows.append((det, t, {"det": det, "p5": round(p5, 1), "chg": round(chg, 1), "fl": fl if fl < 9e9 else None,
                                  "q": b.get("q"), "n": b.get("n"), "sim": b.get("sim")}))
        eng["burn"] = rows
        # 🩳 스퀴즈
        eng["squeeze"] = [(c["score"], c["t"], {"score": c["score"], "si": c.get("si"), "fee": c.get("fee"),
                                                 "st": c.get("st")}) for c in (squeeze or [])]
        # 엔진별 순위 → 종목별 합치기
        cat = {}
        for k, rows in eng.items():
            rows = sorted(rows, key=lambda x: -x[0])[:TOP]
            for rk, (_, t, d) in enumerate(rows, 1):
                cat.setdefault(t, {})[k] = dict(d, rank=rk)
        lst = []
        for t, e in cat.items():
            it = by.get(t) or {}
            r = S.get(t) or {}
            px = it.get("price") or r.get("px")
            if not px:
                continue
            # 예상가: 버닝업 유사 급등 중앙값 → 베스트 유사 사례 중앙값 → 시그널 분포 중앙값
            # 급등이 실제로 나온다는 조건에서의 고점 분포 (유사 사례 중 기준 이상 간 것들)
            q = r.get("cq") or (e.get("burn") or {}).get("q") or r.get("q")
            mid = q[1] if q else None
            sn = it.get("snap") or {}
            warn = _risk(it)
            if sn.get("day_hi") and px < sn["day_hi"] * 0.85:
                warn.append(f"당일 고점 대비 -{(1 - px / sn['day_hi']) * 100:.0f}% 밀림")
            if sn.get("vwap") and px < sn["vwap"] and (sn.get("chg") or 0) > 20:
                warn.append("급등 후 VWAP 아래")
            ranks = [v["rank"] for v in e.values()]
            lst.append({"t": t, "name": it.get("name", ""), "px": px, "exp": px * (1 + mid / 100) if mid else None,
                        "up": mid, "rng": [q[0], q[2]] if q else None, "n": len(e), "avg": round(float(np.mean(ranks)), 1),
                        "eng": e, "warn": warn, "pat": r.get("pat"), "chg": sn.get("chg", r.get("ret1")),
                        "focus": len(e) >= 3 and not warn})
        lst.sort(key=lambda x: (-x["n"], x["avg"]))
        out["hz"][hz] = {"list": lst[:60], "counts": {k: min(len(v), TOP) for k, v in eng.items()}}
    return out


# ── 고정 기록과 적중률 ──
def ledger(cat: dict, prices: dict, path: Path, now_et: datetime, session_day: date) -> dict:
    book = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    have = {(e["day"], e["hz"]) for e in book}
    t_ = now_et.time()
    # 오늘: 본장 시작 15분 뒤 첫 실행에 고정 (진입가 = 그때 가격)
    if now_et.weekday() < 5 and dtime(9, 45) <= t_ < dtime(16, 0) and (session_day.isoformat(), "0") not in have:
        for c in cat["hz"].get("0", {}).get("list", [])[:30]:
            book.append({"day": session_day.isoformat(), "hz": "0", "t": c["t"], "ref": c["px"], "n": c["n"],
                         "eng": {k: v["rank"] for k, v in c["eng"].items()}, "warn": bool(c["warn"]), "res": None})
    # 1거래일: 장 마감 뒤 첫 실행에 고정 (진입가 = 종가) → 다음 거래일 고가로 채점
    if now_et.weekday() < 5 and t_ >= dtime(16, 5) and (session_day.isoformat(), "1") not in have:
        for c in cat["hz"].get("1", {}).get("list", [])[:30]:
            book.append({"day": session_day.isoformat(), "hz": "1", "t": c["t"], "ref": c["px"], "n": c["n"],
                         "eng": {k: v["rank"] for k, v in c["eng"].items()}, "warn": bool(c["warn"]), "res": None})
    # 채점
    for e in book:
        if e["res"] is not None:
            continue
        df = prices.get(e["t"])
        if df is None or not len(df):
            continue
        d0 = date.fromisoformat(e["day"])
        if e["hz"] == "0":
            row = df[[ts.date() == d0 for ts in df.index]]
            if len(row) and (now_et.date() > d0 or t_ >= dtime(16, 30)):
                hi = float(row["High"].iloc[0])
                e["hi"] = round((hi / e["ref"] - 1) * 100, 1)
                e["res"] = e["hi"] >= HIT["0"]
        else:
            nxt = df[[ts.date() > d0 for ts in df.index]]
            if len(nxt) and (nxt.index[0].date() < now_et.date() or t_ >= dtime(16, 30)):
                hi = float(nxt["High"].iloc[0])
                e["hi"] = round((hi / e["ref"] - 1) * 100, 1)
                e["res"] = e["hi"] >= HIT["1"]
    book = book[-6000:]
    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    # 적중률: 엔진별 · 겹친 엔진 수별 (최근 20거래일)
    st = {}
    for hz in ("0", "1"):
        xs = [e for e in book if e["hz"] == hz and e["res"] is not None]
        days = sorted({e["day"] for e in xs})[-20:]
        xs = [e for e in xs if e["day"] in days]
        rate = lambda ys: {"k": sum(y["res"] for y in ys), "n": len(ys),
                           "r": round(sum(y["res"] for y in ys) / len(ys) * 100, 1) if ys else None}
        st[hz] = {"days": len(days), "all": rate(xs),
                  "eng": {k: rate([e for e in xs if k in e["eng"] and e["eng"][k] <= 10]) for k in ENG},
                  "by_n": {str(n): rate([e for e in xs if (e["n"] >= 3 if n == 3 else e["n"] == n)]) for n in (1, 2, 3)},
                  "focus": rate([e for e in xs if e["n"] >= 3 and not e["warn"]]),
                  "th": HIT[hz]}
    return st
