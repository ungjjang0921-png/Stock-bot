"""
소형주 '출발 직전' 감지 (바닥 박스권 → 거래량 동반 이탈 초입)
════════════════════════════════════════════
1) 하루 한 번 셋업 찾기 (일봉, 어제까지):
   - 소형주 ($10 이하 · 시총 3억$ 이하)
   - 6개월 바닥권 (최저가 대비 +60% 이내) · 최근 20일 좁은 박스권
   - 아직 출발 안 함 (최근 20일 하루 +40% 급등 없음)
   - 변동성 수축 + 매집 흔적 (OBV 상승 · 상승일 거래량 우위 · 거래량 서서히 증가)
2) 장중 1분마다 셋업 종목만 확인 → 박스 상단·전일 고가 돌파 + 거래량 유입 시 알림
   - 오늘 +3% ~ +25% 구간에서만 (이미 터진 건 제외 = 추격 방지)
3) 같은 종목은 3일에 한 번 · 증자 뉴스 경고 · 최근 촉매(호재) 있으면 함께 표시
"""

import html
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import catalyst
import guard
import signals

log = logging.getLogger("launch")

# ═════════════════════ 설정 ═════════════════════
MAX_PRICE = 10.0
MAX_MCAP = 300e6
MIN_AVG_VOL = 100_000
UNIVERSE = 600
BOX_DAYS = 20
MAX_BOX = 0.45           # 20일 박스 폭 (고가/저가 -1) 45% 이내
NEAR_LOW = 1.6           # 6개월 최저가의 1.6배 이내 = 바닥권
NO_POP = 40.0            # 최근 20일 하루 +40% 넘게 오른 적 있으면 '이미 출발'로 제외
MAX_SETUPS = 80
MIN_CHG, MAX_CHG = 3.0, 25.0
COOLDOWN_DAYS = 3
DILUTE = ["offering", "registered direct", "private placement", "warrant", "at-the-market",
          "reverse split", "reverse stock split", "priced", "shelf"]
# ═══════════════════════════════════════════════


def find_setup(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 60:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(c.iloc[-1])
    box = df.iloc[-BOX_DAYS:]
    box_hi, box_lo = float(box["High"].max()), float(box["Low"].min())
    if box_lo <= 0 or box_hi / box_lo - 1 > MAX_BOX:
        return None
    low6 = float(l.iloc[-126:].min())
    if price > low6 * NEAR_LOW:
        return None                                              # 바닥권 아님
    day_chg = (c / c.shift(1) - 1) * 100
    if float(day_chg.iloc[-BOX_DAYS:].max()) >= NO_POP:
        return None                                              # 이미 출발함
    up = c.diff() > 0
    sign = np.sign(c.diff().fillna(0))
    obv = (sign * v).cumsum()
    obv_up = float(obv.iloc[-1] - obv.iloc[-BOX_DAYS]) > 0
    upv = float(v.iloc[-BOX_DAYS:][up.iloc[-BOX_DAYS:]].sum())
    dnv = float(v.iloc[-BOX_DAYS:][~up.iloc[-BOX_DAYS:]].sum())
    vr = float(v.iloc[-5:].mean() / max(v.iloc[-BOX_DAYS:].mean(), 1))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    squeeze = float(tr.iloc[-10:].mean() / max(tr.iloc[-60:].mean(), 1e-9))
    hits = []
    score = 0.0
    if obv_up:
        score += 20; hits.append("OBV 상승(매집)")
    if dnv > 0 and upv / dnv >= 1.2:
        score += 15; hits.append(f"상승일 거래량 {upv / dnv:.1f}배")
    if 1.1 <= vr <= 4:
        score += 15; hits.append(f"최근 거래량 증가 {vr:.1f}배")
    if squeeze < 0.8:
        score += 15; hits.append("변동성 수축")
    if price >= box_hi * 0.9:
        score += 15; hits.append("박스 상단 근접")
    score += max(0, 20 - (price / low6 - 1) * 40)                # 바닥에 가까울수록
    if score < 40:
        return None
    hi6 = float(h.iloc[-126:].max())
    return {"box_hi": box_hi, "box_lo": box_lo, "low6": low6, "hi6": hi6, "prev_high": float(h.iloc[-1]),
            "prev_close": price, "avgvol": float(v.iloc[-20:].mean()), "score": round(score), "hits": hits}


def refresh(state: dict) -> None:
    rows = catalyst.tv_query({
        "markets": ["america"],
        "filter": [{"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                   {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                   {"left": "close", "operation": "in_range", "right": [0.1, MAX_PRICE]},
                   {"left": "market_cap_basic", "operation": "less", "right": MAX_MCAP},
                   {"left": "average_volume_10d_calc", "operation": "greater", "right": MIN_AVG_VOL}],
        "columns": ["description", "close", "float_shares_outstanding"],
        "sort": {"sortBy": "average_volume_10d_calc", "sortOrder": "desc"}, "range": [0, UNIVERSE]})
    info = {r["t"]: r for r in rows}
    today = datetime.now(catalyst.ETZ).date()
    found = {}
    tickers = list(info)
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            data = yf.download(chunk, period="7mo", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            log.warning("일봉 다운로드 실패: %s", e)
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["Close"])
                df = df[[ts.date() < today for ts in df.index]]
                s = find_setup(df)
                if s:
                    s["float"] = catalyst._num(info[t].get("float_shares_outstanding"))
                    found[t] = s
            except Exception:
                continue
    state["launch_setups"] = dict(sorted(found.items(), key=lambda kv: -kv[1]["score"])[:MAX_SETUPS])
    state["launch_day"] = today.isoformat()
    log.info("출발 셋업 %d개 (후보 %d 중)", len(state["launch_setups"]), len(tickers))


def _dilution_news(t: str) -> str | None:
    try:
        for it in (yf.Ticker(t).news or [])[:10]:
            title = ((it.get("content") or it).get("title") or "")
            if any(w in title.lower() for w in DILUTE):
                return title
    except Exception:
        pass
    return None


def scan(state: dict, send) -> None:
    now_et = datetime.now(catalyst.ETZ)
    f = catalyst.session_fields(now_et)
    if not f:
        return
    if state.get("launch_day") != now_et.date().isoformat():
        refresh(state)
    setups = state.get("launch_setups") or {}
    alerted = state.setdefault("launch_alerted", {})
    for t in [t for t, d in alerted.items()
              if now_et.date() - datetime.fromisoformat(d).date() > timedelta(days=COOLDOWN_DAYS)]:
        del alerted[t]
    todo = [t for t in setups if t not in alerted]
    if not todo:
        return
    q = catalyst.quotes(todo)
    sess = f[3]
    recent = state.get("cat_recent", {})
    for t in todo:
        qt, s = q.get(t), setups[t]
        if not qt:
            continue
        price, chg, vol = catalyst.live(qt, now_et)
        if not price or chg is None or not (MIN_CHG <= chg <= MAX_CHG):
            continue
        need = s["avgvol"] * {"프리장": 0.15, "정규장": 0.6, "애프터장": 0.3}.get(sess, 0.5)
        if not vol or vol < need:
            continue
        if price > s["box_hi"]:
            why = f"20일 박스 상단 {catalyst._p(s['box_hi'])} 돌파"
        elif price > s["prev_high"]:
            why = f"전일 고가 {catalyst._p(s['prev_high'])} 돌파 (박스 상단 {catalyst._p(s['box_hi'])}까지 {(s['box_hi'] / price - 1) * 100:+.0f}%)"
        else:
            continue
        ok, bad, ginfo = guard.pre_check(t, price, chg)
        if not ok:
            continue                                            # 설거지 위험 → 알림 안 보냄 (15분 뒤 재검사)
        alerted[t] = now_et.date().isoformat()
        height = s["box_hi"] - s["box_lo"]
        t1 = max(s["box_hi"] + height, price * 1.15)
        t2 = max(s["hi6"], t1 * 1.2)
        stop = max(s["box_lo"], (s["box_hi"] + s["box_lo"]) / 2 * 0.97)
        rel = vol / s["avgvol"] if s["avgvol"] else 0
        cat = recent.get(t)
        warn = _dilution_news(t)
        up = "위에 매물 없음" if ginfo.get("upside") is None else f"다음 저항 +{ginfo['upside']}%"
        signals.buy(state, send, t, price, "출발",
                    why=[why, f"바닥 박스 {catalyst._p(s['box_lo'])}~{catalyst._p(s['box_hi'])}",
                         f"거래량 평소 {rel:.1f}배", up, f"재료 [{cat['grade']}] {cat['label']}" if cat else ""],
                    name=qt.get("description") or "", dilute=bool(warn))
        catalyst.log_alert(state, "출발", t, price, score=s["score"])


def list_text(state: dict) -> str:
    setups = state.get("launch_setups") or {}
    if not setups:
        return "🛫 출발 대기 후보가 아직 없어요. (미국장 시간에 하루 한 번 새로 찾아요)"
    rows = [f"{t:<5} {catalyst._p(s['prev_close']):>8} {catalyst._p(s['box_hi']):>8} {s['score']:>3}"
            for t, s in list(setups.items())[:25]]
    body = "티커       종가  돌파가격 점수\n" + "─" * 30 + "\n" + "\n".join(rows)
    return (f"🛫 <b>출발 대기 후보</b> ({state.get('launch_day')} · {len(setups)}개)\n<pre>{html.escape(body)}</pre>"
            "돌파가격을 거래량과 함께 넘으면 🛫 알림이 와요.")
