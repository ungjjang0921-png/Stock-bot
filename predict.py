"""
급등 예측 (학습 판별 모델 · 성능 비교 · 고정 예측 기록)
════════════════════════════════════════════
- 목표: 기준일 종가 대비 N거래일 안에 고가가 +X% 이상 (다음 거래일 +20% · 5거래일 +30% · 10거래일 +40%)
- 기존 방식: 비슷한 조건(거래량 배수·등락·주가대) 과거 빈도
- 개선 모델: 17개 차트·수급 특징으로 학습한 로지스틱 판별 모델 (급등 성공·실패 비교)
- 검증: 시간순 분리 — 최근 60거래일은 학습에 안 쓰고, 그 기간 매일 상위 10개를 골랐을 때 실제 적중률 비교
- 확률: 검증 구간에서 점수대별 표본이 충분할 때만 표시 (아니면 '확률 미제시')
- 고정 예측 기록: 매 거래일 상위 후보를 docs/pred_log.json 에 고정 저장 → 기간이 지나면 실제 결과로 채점
"""

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("predict")

HORIZONS = {1: 0.20, 5: 0.30, 10: 0.40}      # 거래일: 목표 상승률
EVAL_DAYS = 60
PICK_PER_DAY = 10
CAND_TOP = 30
LOG_TOP = 20
FEATS = ["ret1", "ret5", "ret20", "relvol", "vol20", "range20", "dlow", "dhigh", "ma20", "ma50", "rsi", "cp",
         "updown", "obv", "logp", "dvol", "gap", "rsi20", "rsimin5", "pump"]
FEAT_KO = {"ret1": "전일 등락", "ret5": "5일 수익률", "ret20": "20일 수익률", "relvol": "거래량 배수",
           "vol20": "변동성", "range20": "20일 고저폭", "dlow": "6개월 저점 대비", "dhigh": "6개월 고점 대비",
           "ma20": "20일선 이격", "ma50": "50일선 이격", "rsi": "RSI", "cp": "종가 위치", "updown": "상승일 거래량 우위",
           "obv": "OBV 기울기", "logp": "주가 수준", "dvol": "거래대금", "gap": "시가 갭",
           "rsi20": "RSI 20 근처", "rsimin5": "최근 5일 RSI 저점", "pump": "과거 급등 크기"}
RSI_ZONE = (12, 28)      # 'RSI 20 근처' 구간


def features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"].astype(float)
    r1 = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret1"] = r1
    f["ret5"] = c / c.shift(5) - 1
    f["ret20"] = c / c.shift(20) - 1
    f["relvol"] = np.log1p(v / v.rolling(20).mean().shift(1))
    f["vol20"] = r1.rolling(20).std()
    f["range20"] = np.log(h.rolling(20).max() / l.rolling(20).min())
    f["dlow"] = np.log(c / l.rolling(126, min_periods=40).min())
    f["dhigh"] = np.log(c / h.rolling(126, min_periods=40).max())
    f["ma20"] = c / c.rolling(20).mean() - 1
    f["ma50"] = c / c.rolling(50).mean() - 1
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = (100 - 100 / (1 + g / ls.replace(0, np.nan))).fillna(50) / 100
    f["cp"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    up = d > 0
    f["updown"] = np.log((v * up).rolling(10).sum() + 1) - np.log((v * ~up).rolling(10).sum() + 1)
    obv = (np.sign(d.fillna(0)) * v).cumsum()
    f["obv"] = (obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)
    f["logp"] = np.log(c.clip(lower=0.01))
    f["dvol"] = np.log1p((c * v).rolling(20).mean())
    f["gap"] = o / c.shift(1) - 1
    rsi100 = f["rsi"] * 100
    f["rsi20"] = np.exp(-((rsi100 - 20) / 7) ** 2)                 # RSI 20에 가까울수록 1
    f["rsimin5"] = f["rsi"].rolling(5).min()
    f["pump"] = np.log((h / c.shift(1)).rolling(250, min_periods=60).max())   # 지난 1년 하루 최대 급등폭
    for H in HORIZONS:
        fh = h[::-1].rolling(H, min_periods=H).max()[::-1].shift(-1)
        f[f"g{H}"] = fh / c - 1
        hv = h.values.astype(float)
        dpk = np.full(len(df), np.nan)
        if len(df) > H:
            win = np.lib.stride_tricks.sliding_window_view(hv[1:], H)      # i 다음날부터 H일
            dpk[:len(win)] = np.argmax(win, axis=1) + 1
            dpk[len(df) - H:] = np.nan
        f[f"d{H}"] = dpk
    f["close"] = c
    f["vbin"] = np.log1p(v / v.rolling(20).mean().shift(1))
    return f.replace([np.inf, -np.inf], np.nan)


def _logit_fit(X, y, l2=1e-3, iters=250, lr=0.05, weights=None):
    n, k = X.shape
    w = np.zeros(k); b = 0.0
    pos = y.mean()
    sw = np.where(y == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))   # 급등·비급등 균형
    if weights is not None:
        sw = sw * weights / weights.mean()
    m_w = np.zeros(k); v_w = np.zeros(k); m_b = v_b = 0.0
    for t in range(1, iters + 1):
        z = np.clip(X @ w + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        e = (p - y) * sw
        gw = X.T @ e / n + l2 * w
        gb = e.mean()
        m_w = 0.9 * m_w + 0.1 * gw; v_w = 0.999 * v_w + 0.001 * gw * gw
        m_b = 0.9 * m_b + 0.1 * gb; v_b = 0.999 * v_b + 0.001 * gb * gb
        w -= lr * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
    return w, b


MIN_BUCKET = 60          # 확률 표시에 필요한 검증 표본 수
MIN_SURE = 20.0          # 확실 후보: 검증 적중률 95% 하한이 이 값(%) 이상이어야 함


def _gbm(X, y):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    n = len(y)
    idx = np.random.default_rng(0).choice(n, size=min(n, 300_000), replace=False)
    pos = y[idx].mean()
    sw = np.where(y[idx] == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    m.fit(X[idx], y[idx], sample_weight=sw)
    return m


def _peak_model(X, y):
    """급등했을 때 고가 상승폭(log) 예측 — 저유통·과거 급등 큰 종목은 더 크게"""
    if len(y) < 200:
        return None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=0.06,
                                          min_samples_leaf=50, random_state=0)
        m.fit(X, y)
        return m.predict
    except ImportError:
        A = np.c_[X, np.ones(len(X))]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return lambda Z: np.c_[Z, np.ones(len(Z))] @ coef


def _wilson(k, n):
    if not n:
        return None, None
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - m) * 100, 1), round(min(1, c + m) * 100, 1)


def _base_key(f):
    rv = np.digitize(np.expm1(f["vbin"].fillna(0)), [1.5, 3, 10])
    ch = np.digitize(f["ret1"].fillna(0) * 100, [-10, -3, 3, 10, 30])
    px = np.digitize(f["close"].fillna(1), [0.5, 1, 5])
    return rv * 100 + ch * 10 + px


def _evaluate(ev, score, H, th, gcol, dcol, pk_med, dd_med, pk_pred=None):
    rows = []
    ev = ev.assign(s=score, pk=pk_pred if pk_pred is not None else pk_med)
    for dte, g in ev.groupby(level="date"):
        rows.append(g.nlargest(PICK_PER_DAY, "s"))
    if not rows:
        return {}
    sel = pd.concat(rows)
    hit = sel[gcol] >= th
    k, n = int(hit.sum()), int(len(sel))
    lo, hi = _wilson(k, n)
    hits = sel[hit]
    pk_err = float((hits[gcol] - hits["pk"]).abs().mean() * 100) if len(hits) else None
    dd_err = float((hits[dcol] - dd_med).abs().mean()) if len(hits) else None
    return {"rate": round(k / n * 100, 1), "k": k, "n": n, "lo": lo, "hi": hi,
            "pk_err": None if pk_err is None else round(pk_err, 1), "dd_err": None if dd_err is None else round(dd_err, 1),
            "nh": int(len(hits))}


def run(prices: dict[str, pd.DataFrame], info: dict[str, dict], cutoff: date, log_path: Path,
        fx: float | None) -> dict:
    frames = []
    for t, df in prices.items():
        if df is None or len(df) < 80:
            continue
        d = df[[ts.date() < cutoff for ts in df.index]]
        if len(d) < 80:
            continue
        f = features(d)
        f["t"] = t
        f["date"] = [ts.date() for ts in f.index]
        frames.append(f.reset_index(drop=True))
    if not frames:
        return {}
    A = pd.concat(frames, ignore_index=True).set_index(["date", "t"]).sort_index()
    A = A.dropna(subset=FEATS)
    dates = sorted(A.index.get_level_values("date").unique())
    asof = dates[-1]
    out = {"asof": asof.isoformat(), "horizons": {}, "fx": fx}
    latest = A.xs(asof, level="date")
    for H, th in HORIZONS.items():
        gcol, dcol = f"g{H}", f"d{H}"
        lab = A.dropna(subset=[gcol])
        ldates = sorted(lab.index.get_level_values("date").unique())
        if len(ldates) < EVAL_DAYS + 100:
            log.warning("%d거래일 예측: 데이터 부족 (%d일)", H, len(ldates))
            continue
        ev_start = ldates[-EVAL_DAYS]
        tr = lab[lab.index.get_level_values("date") < ev_start - timedelta(days=H * 2)]
        ev = lab[lab.index.get_level_values("date") >= ev_start]
        Xtr = tr[FEATS].clip(tr[FEATS].quantile(0.005), tr[FEATS].quantile(0.995), axis=1)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        norm = lambda X: ((X.clip(tr[FEATS].quantile(0.005), tr[FEATS].quantile(0.995), axis=1) - mu) / sd).fillna(0).values
        ytr = (tr[gcol] >= th).astype(float).values
        Xn_tr, Xn_ev, Xn_now = norm(tr[FEATS]), norm(ev[FEATS]), norm(latest[FEATS])
        w, b = _logit_fit(Xn_tr, ytr)
        models = {"로지스틱": (Xn_ev @ w + b, Xn_now @ w + b)}
        # RSI 20 근처 가중 모델 (ATM·상폐 위험 종목은 아래에서 제외)
        rw_ = 1 + 2 * tr["rsi20"].values
        w2, b2 = _logit_fit(Xn_tr, ytr, weights=rw_)
        models["RSI20 가중"] = (Xn_ev @ w2 + b2, Xn_now @ w2 + b2)
        gb = _gbm(Xn_tr, ytr)                                   # 비선형 모델 (설치돼 있으면)
        if gb is not None:
            models["부스팅"] = (gb.predict_proba(Xn_ev)[:, 1], gb.predict_proba(Xn_now)[:, 1])
        # 기존 방식: 같은 조건 과거 빈도
        tk = _base_key(tr)
        freq = pd.Series(ytr).groupby(tk.values if hasattr(tk, "values") else tk).mean()
        b_ev = pd.Series(_base_key(ev)).map(freq).fillna(ytr.mean()).values
        surge_tr = tr[tr[gcol] >= th]
        pk_med = float(surge_tr[gcol].median()) if len(surge_tr) else th
        dd_med = float(surge_tr[dcol].median()) if len(surge_tr) else H
        # 급등 크기 예측: 급등한 날만 모아 '얼마나 크게' 갔는지 학습 (종목별 조건부 고점)
        reg = _peak_model(Xn_tr[ytr == 1], np.log1p(tr[gcol].values[ytr == 1].clip(0, 20)))
        pk_ev = np.expm1(reg(Xn_ev)) if reg else None
        pk_now = np.expm1(reg(Xn_now)) if reg else np.full(len(Xn_now), pk_med)
        big_tr = surge_tr[surge_tr[gcol] >= 2 * th]
        big_ratio = round(len(big_tr) / max(len(surge_tr), 1) * 100, 1)
        old = _evaluate(ev, b_ev, H, th, gcol, dcol, pk_med, dd_med)
        res = {k: _evaluate(ev, v[0], H, th, gcol, dcol, pk_med, dd_med, pk_ev) for k, v in models.items()}
        # RSI 20 근처 근거 (5년 학습 데이터)
        rz = tr["rsi"].between(RSI_ZONE[0] / 100, RSI_ZONE[1] / 100)
        rsi_stat = {"zone": round(float(ytr[rz.values].mean()) * 100, 2) if rz.any() else None,
                    "all": round(float(ytr.mean()) * 100, 2), "n": int(rz.sum())}
        best = max(res, key=lambda k: res[k].get("rate", 0))       # 검증 적중률 높은 모델 자동 선택
        new, (s_ev, s_now) = res[best], models[best]
        base_rate = round(float((ev[gcol] >= th).mean()) * 100, 1)
        # 점수 → 백분위 · 점수대별 검증 적중률(표본 충분할 때만 확률 표시)
        srt = np.sort(s_ev)
        pct = lambda s: np.searchsorted(srt, s) / max(len(s_ev), 1) * 100
        ev_p = pct(s_ev)
        yev = ev[gcol].values >= th
        cal = {}
        for lo_ in (99.8, 99.5, 99, 98, 97, 95, 90, 80):
            m = ev_p >= lo_
            if m.sum() >= MIN_BUCKET:
                k_ = int(yev[m].sum()); n_ = int(m.sum())
                lo95, hi95 = _wilson(k_, n_)
                cal[lo_] = {"p": round(k_ / n_ * 100, 1), "n": n_, "lo": lo95}
        # 확실 후보 기준: 95% 하한이 무작위의 2배 이상인 점수대 중 가장 높은 적중률
        sure_cut = None
        ok = [(k, v) for k, v in cal.items() if v["lo"] is not None and v["lo"] >= max(2 * base_rate, MIN_SURE)]
        if ok:
            sure_cut = max(ok, key=lambda kv: (kv[1]["p"], -kv[0]))[0]
        p_now = pct(s_now)
        order = np.argsort(-s_now)[:CAND_TOP]
        cands = []
        for i in order:
            t = latest.index[i]
            px = float(latest["close"].iloc[i])
            sc = round(float(p_now[i]), 1)
            bk = next((k for k in sorted(cal, reverse=True) if sc >= k), None)
            prob = cal[bk]["p"] if bk is not None else None
            risk = (info.get(t) or {}).get("risk") or []
            sure = sure_cut is not None and sc >= sure_cut and not risk
            top = sorted(zip(FEATS, (Xn_now[i] * w)), key=lambda x: -x[1])[:3]
            rsi_v = float(latest["rsi"].iloc[i]) * 100
            up_i = float(max(th, min(pk_now[i], 20)))
            cands.append({"t": t, "name": (info.get(t) or {}).get("name", ""), "px": px,
                          "chg": round(float(latest["ret1"].iloc[i]) * 100, 1), "score": sc, "prob": prob,
                          "peak": px * (1 + up_i), "up": round(up_i * 100, 1), "sure": sure, "risk": risk,
                          "rsi": round(rsi_v, 1),
                          "rsi20": RSI_ZONE[0] <= rsi_v <= RSI_ZONE[1] and not risk,
                          "day": _add_bd(asof, int(round(dd_med))).isoformat(),
                          "why": [FEAT_KO[f] for f, v in top if v > 0]})
        cands.sort(key=lambda c: (not c["sure"], bool(c["risk"]), not c["rsi20"], -c["score"]))
        imp = sorted(zip(FEATS, w), key=lambda x: -abs(x[1]))[:6]
        out["horizons"][str(H)] = {
            "th": int(th * 100), "old": old, "new": new, "model": best, "models": res, "base": base_rate,
            "cands": cands, "cal": {str(k): v for k, v in cal.items()},
            "sure_cut": sure_cut, "sure": cal.get(sure_cut) if sure_cut is not None else None,
            "pk_med": round(pk_med * 100, 1), "dd_med": round(dd_med, 1), "big": big_ratio, "rsi_stat": rsi_stat,
            "imp": [{"f": FEAT_KO[f], "w": round(float(v), 2)} for f, v in imp],
            "train_n": int(len(tr)), "train_pos": int(ytr.sum()), "eval_from": ev_start.isoformat()}
    out["ledger"] = ledger(out, prices, log_path)
    return out


def _add_bd(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def ledger(out: dict, prices: dict, path: Path) -> list[dict]:
    """매 거래일 상위 후보를 고정 저장하고, 기간이 지나면 실제 고점으로 채점"""
    try:
        book = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        book = []
    have = {(e["asof"], e["h"], e["t"]) for e in book}
    for H, hz in out.get("horizons", {}).items():
        for rk, cnd in enumerate(hz["cands"][:LOG_TOP], 1):
            cnd_sure = cnd.get("sure", False)
            k = (out["asof"], int(H), cnd["t"])
            if k not in have:
                book.append({"asof": out["asof"], "h": int(H), "t": cnd["t"], "name": cnd["name"], "rank": rk,
                             "ref": cnd["px"], "peak": cnd["peak"], "day": cnd["day"], "th": hz["th"],
                             "score": cnd["score"], "sure": cnd_sure, "res": None, "act_peak": None, "act_day": None})
    for e in book:
        if e["res"] in ("적중", "미적중"):
            continue
        df = prices.get(e["t"])
        if df is None:
            continue
        a = date.fromisoformat(e["asof"])
        after = df[[ts.date() > a for ts in df.index]]
        if not len(after):
            continue
        win = after.iloc[:e["h"]]
        hi = float(win["High"].max())
        e["act_peak"] = hi
        e["act_day"] = win["High"].idxmax().date().isoformat()
        if hi >= e["ref"] * (1 + e["th"] / 100):
            e["res"] = "적중"
        elif len(win) >= e["h"]:
            e["res"] = "미적중"
        else:
            e["res"] = "관찰 중"
    book = sorted(book, key=lambda e: (e["asof"], e["h"], e["rank"]))[-3000:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    res = {}
    for H in HORIZONS:
        xs = [e for e in book if e["h"] == H and e["res"] in ("적중", "미적중")]
        if xs:
            k = sum(e["res"] == "적중" for e in xs)
            lo, hi = _wilson(k, len(xs))
            res[str(H)] = {"k": k, "n": len(xs), "rate": round(k / len(xs) * 100, 1), "lo": lo, "hi": hi}
            ss = [e for e in xs if e.get("sure")]
            if ss:
                ks = sum(e["res"] == "적중" for e in ss)
                res[str(H)]["sure"] = {"k": ks, "n": len(ss), "rate": round(ks / len(ss) * 100, 1)}
    out["ledger_stat"] = res
    return [e for e in reversed(book)][:150]
