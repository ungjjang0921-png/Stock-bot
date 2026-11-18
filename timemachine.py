"""⏱ 타임머신 — 오늘 장중 흐름과 비슷했던 과거 장중 사례를 찾아서
'다음 저점 시각·가격'과 '그 뒤 고점 시각·가격'을 예측하는 장중 단타 도구.

- 5분봉(프리·정규·애프터, 최근 60일)을 미국 동부 04:00~20:00 기준 192칸으로 맞춤
- 지금 시각까지의 흐름(최근 3시간 모양 · 전일 종가 대비 위치 · 당일 고점 대비 · 누적 거래량)이
  비슷한 과거 날을 같은 종목 우선으로 12개 고름
- 그 날들에서 '지금 시각 이후' 최저가와, 최저가 이후 최고가가 언제·얼마였는지 모아서
  중앙값(대표)과 25~75% 범위(50% 구간)를 보여줌
- 과거 검증: 최근 날들을 하나씩 '그날 이전 데이터만으로' 같은 시각에 예측해 보고
  저점 지정가 체결률 · 목표 도달률 · 1회 평균 순수익(왕복 수수료 반영)을 매김
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("timemachine")
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

SLOT = 5                 # 분
NS = 192                 # 04:00~20:00 ET
LOOK = 36                # 최근 3시간 모양 비교
K = 12                   # 비슷한 날 개수
FEE = 0.5                # 왕복 수수료·슬리피지 %
SAME_BONUS = 0.7         # 같은 종목이면 거리 × 0.7 (우선)
VAL_DAYS = 25            # 검증에 쓰는 최근 날 수


def _grid(df: pd.DataFrame) -> dict:
    """5분봉 → {날짜: (C,H,L,V, prev_close)} — 칸마다 로그(가격/전일 종가)"""
    if df is None or df.empty:
        return {}
    idx = df.index.tz_convert(ET) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(ET)
    df = df.set_axis(idx)
    slot = ((idx.hour * 60 + idx.minute) - 240) // SLOT
    df = df.assign(_s=slot, _d=idx.date)
    df = df[(df["_s"] >= 0) & (df["_s"] < NS)]
    out, prev = {}, None
    for d, g in df.groupby("_d"):
        reg = g[(g["_s"] >= 66) & (g["_s"] < 144)]                  # 09:30~16:00
        if prev is not None and prev > 0:
            C = np.full(NS, np.nan); H = np.full(NS, np.nan); L = np.full(NS, np.nan); V = np.zeros(NS)
            s = g["_s"].values
            C[s] = g["Close"].values; H[s] = g["High"].values; L[s] = g["Low"].values
            V[s] = np.nan_to_num(g["Volume"].values.astype(float))
            C = pd.Series(C).ffill().fillna(prev).values              # 거래 없는 칸은 직전 가격
            H = np.where(np.isnan(H), C, H); L = np.where(np.isnan(L), C, L)
            out[d] = {"c": np.log(C / prev), "h": np.log(H / prev), "l": np.log(L / prev), "v": V,
                      "prev": prev, "n": int(len(g))}
        if len(reg):
            prev = float(reg["Close"].iloc[-1])
    return out


def _feat(day: dict, s: int, avgv: float) -> tuple:
    c = day["c"]
    a = max(0, s - LOOK + 1)
    shape = c[a:s + 1] - c[s]
    if len(shape) < LOOK:
        shape = np.concatenate([np.full(LOOK - len(shape), shape[0]), shape])
    hi = np.max(day["h"][:s + 1]) - c[s]
    lo = c[s] - np.min(day["l"][:s + 1])
    vol = np.log1p(day["v"][:s + 1].sum() / max(avgv, 1))
    return shape, np.array([c[s], hi, lo, vol])


def _future(day: dict, s: int) -> dict | None:
    """지금 칸(s) 이후: 최저가(칸·로그수익), 그 뒤 최고가"""
    if s >= NS - 2:
        return None
    c0 = day["c"][s]
    l, h = day["l"][s + 1:], day["h"][s + 1:]
    i = int(np.argmin(l))
    j = i + int(np.argmax(h[i:]))
    return {"ls": s + 1 + i, "lr": float(l[i] - c0), "hs": s + 1 + j, "hr": float(h[j] - l[i]),
            "end": float(day["c"][-1] - c0), "lo": l - c0, "hi": h - c0}


def _neighbors(q_shape, q_vec, pool, t, sd):
    if not pool:
        return []
    S = np.stack([p["shape"] for p in pool]); X = np.stack([p["vec"] for p in pool])
    d = ((S - q_shape) ** 2).mean(1) / sd[0] + (((X - q_vec) ** 2) / sd[1]).sum(1) * 0.5
    d = d * np.array([SAME_BONUS if p["t"] == t else 1.0 for p in pool])
    o = np.argsort(d)[:K]
    return [dict(pool[i], dist=float(d[i])) for i in o]


def _summ(nb: list) -> dict | None:
    F = [p["fut"] for p in nb if p["fut"]]
    if len(F) < 5:
        return None
    ls = np.array([f["ls"] for f in F]); lr = np.array([f["lr"] for f in F])
    hs = np.array([f["hs"] for f in F]); hr = np.array([f["hr"] for f in F])
    q = lambda a: [float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75))]
    combo = {}
    for f in F:
        k = (f["ls"] * SLOT // 60, f["hs"] * SLOT // 60)
        combo[k] = combo.get(k, 0) + 1
    best = max(combo.values())
    return {"ls": q(ls), "lr": q(lr), "hs": q(hs), "hr": q(hr), "n": len(F), "combo": best,
            "same": sum(1 for p in nb if p.get("same"))}


def _slot_time(day, s: int) -> datetime:
    return datetime(day.year, day.month, day.day, 4, 0, tzinfo=ET) + timedelta(minutes=s * SLOT)


def build(frames: dict, now_et: datetime, names: dict | None = None) -> dict:
    """frames: {티커: 5분봉 DataFrame(60일·프리포스트)}"""
    names = names or {}
    G = {t: _grid(df) for t, df in frames.items()}
    G = {t: g for t, g in G.items() if len(g) >= 10}
    today = now_et.date()
    s_now = ((now_et.hour * 60 + now_et.minute) - 240) // SLOT
    out = {"at": now_et.astimezone(KST).strftime("%m/%d %H:%M"), "rows": [], "fee": FEE, "k": K}
    if not (0 <= s_now < NS - 4):
        out["closed"] = True
        return out
    # 과거 날 풀 (오늘 제외)
    avgv = {t: float(np.median([d["v"].sum() for dd, d in g.items() if dd < today] or [1])) for t, g in G.items()}

    def pool_at(s, before):
        P = []
        for t, g in G.items():
            for dd, d in g.items():
                if dd >= before:
                    continue
                sh, vec = _feat(d, s, avgv[t])
                P.append({"t": t, "d": dd, "shape": sh, "vec": vec, "fut": _future(d, s)})
        return P

    P = pool_at(s_now, today)
    if len(P) < 30:
        return out
    S = np.stack([p["shape"] for p in P]); X = np.stack([p["vec"] for p in P])
    sd = (max(float(S.var()), 1e-5), np.maximum(X.var(0), 1e-5))

    # ── 오늘 예측 ──
    for t, g in G.items():
        d = g.get(today)
        if not d or d["n"] < 3:
            continue
        sh, vec = _feat(d, s_now, avgv[t])
        nb = _neighbors(sh, vec, P, t, sd)
        for p in nb:
            p["same"] = p["t"] == t
        sm = _summ(nb)
        if not sm:
            continue
        px = float(d["prev"] * np.exp(d["c"][s_now]))
        lo_px = [px * np.exp(x) for x in sm["lr"]]
        hi_px = [lo_px[1] * np.exp(x) for x in sm["hr"]]
        lt = [_slot_time(today, int(round(x))) for x in sm["ls"]]
        ht = [_slot_time(today, int(round(x))) for x in sm["hs"]]
        k_ = lambda x: x.astimezone(KST).strftime("%m/%d %H:%M")
        wide = (sm["ls"][2] - sm["ls"][0]) * SLOT >= 120 or (sm["hs"][2] - sm["hs"][0]) * SLOT >= 180
        notes = ["장중 예측"]
        if wide:
            notes.append("시간 분산 · 대표 시나리오")
        if sm["combo"] <= 2:
            notes.append("시간대 반복 표본 적음")
        rel_v = float(np.log1p(d["v"][:s_now + 1].sum() / max(avgv[t], 1)))
        out["rows"].append({
            "t": t, "name": names.get(t, ""), "px": round(px, 4), "chg": round((np.exp(d["c"][s_now]) - 1) * 100, 1),
            "low": {"px": round(lo_px[1], 4), "r": [round(lo_px[0], 4), round(lo_px[2], 4)],
                    "at": k_(lt[1]), "ar": [k_(lt[0])[6:], k_(lt[2])[6:]]},
            "high": {"px": round(hi_px[1], 4), "r": [round(hi_px[0], 4), round(hi_px[2], 4)],
                     "at": k_(ht[1]), "ar": [k_(ht[0])[6:], k_(ht[2])[6:]]},
            "to_low": round((np.exp(sm["lr"][1]) - 1) * 100, 2),
            "lo_hi": round((np.exp(sm["hr"][1]) - 1) * 100, 2),
            "ret_low": round((np.exp(sm["hr"][1]) - 1) * 100 - FEE, 2),
            "ret_now": round((np.exp(sm["lr"][1] + sm["hr"][1]) - 1) * 100 - FEE, 2),
            "combo": sm["combo"], "n": sm["n"], "same": sm["same"], "notes": notes, "rv": round(float(np.expm1(rel_v)), 2),
            "soon": round((sm["ls"][1] - s_now) * SLOT),
            "an": [{"t": p["t"], "d": p["d"].strftime("%m/%d"), "lt": k_(_slot_time(p["d"], p["fut"]["ls"]))[6:],
                    "ht": k_(_slot_time(p["d"], p["fut"]["hs"]))[6:],
                    "lr": round((np.exp(p["fut"]["lr"]) - 1) * 100, 1), "hr": round((np.exp(p["fut"]["hr"]) - 1) * 100, 1)}
                   for p in nb if p["fut"]][:K],
        })
    out["rows"].sort(key=lambda r: -r["ret_now"])
    try:
        out["val"] = validate(G, avgv, s_now, today)
    except Exception as e:
        log.warning("타임머신 검증 실패: %s", e)
    log.info("⏱ 타임머신 %d종목 · 검증 %s", len(out["rows"]), out.get("val"))
    return out


def validate(G, avgv, s, today) -> dict | None:
    """최근 날마다 '그 전 데이터만'으로 같은 시각에 예측 → 저점 지정가 매수 · 예측 고점 매도 · 못 가면 그날 마지막 가격에 정리"""
    days = sorted({dd for g in G.values() for dd in g if dd < today})[-VAL_DAYS:]
    if len(days) < 5:
        return None
    allP = []
    for t, g in G.items():
        for dd, d in g.items():
            if dd >= today:
                continue
            sh, vec = _feat(d, s, avgv[t])
            allP.append({"t": t, "d": dd, "shape": sh, "vec": vec, "fut": _future(d, s), "day": d})
    S = np.stack([p["shape"] for p in allP]); X = np.stack([p["vec"] for p in allP])
    sd = (max(float(S.var()), 1e-5), np.maximum(X.var(0), 1e-5))
    n = fill = win = 0
    rets, rets_now, in_t = [], [], 0
    for q in allP:
        if q["d"] not in days or not q["fut"] or q["day"]["n"] < 20:
            continue
        pool = [p for p in allP if p["d"] < q["d"]]
        if len(pool) < 30:
            continue
        sm = _summ(_neighbors(q["shape"], q["vec"], pool, q["t"], sd))
        if not sm:
            continue
        n += 1
        f = q["fut"]
        lo_t, hi_t = sm["lr"][1], sm["hr"][1]
        if sm["ls"][0] <= f["ls"] <= sm["ls"][2]:
            in_t += 1
        # 현재가 매수 → 예측 고점(lo_t+hi_t) 또는 그날 마지막 가격
        tgt_now = lo_t + hi_t
        rets_now.append((np.exp(tgt_now) - 1) * 100 - FEE if np.max(f["hi"]) >= tgt_now else (np.exp(f["end"]) - 1) * 100 - FEE)
        # 저점 지정가
        hit = np.where(f["lo"] <= lo_t)[0]
        if not len(hit):
            continue
        fill += 1
        i = int(hit[0])
        if np.max(f["hi"][i:]) >= lo_t + hi_t:
            win += 1
            rets.append((np.exp(hi_t) - 1) * 100 - FEE)
        else:
            rets.append((np.exp(f["end"] - lo_t) - 1) * 100 - FEE)
    if n < 20:
        return None
    md = lambda a: round(float(np.mean(a)), 2) if a else None
    return {"n": n, "days": len(days), "fill": round(fill / n * 100, 1), "win": round(win / max(fill, 1) * 100, 1),
            "avg": md(rets), "med": round(float(np.median(rets)), 2) if rets else None,
            "avg_now": md(rets_now), "pos_now": round(sum(1 for x in rets_now if x > 0) / max(len(rets_now), 1) * 100, 1),
            "time_in": round(in_t / n * 100, 1)}
