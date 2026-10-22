"""
설거지 필터 + 추천 후 감시
════════════════════════════════════════════
pre_check(): 알림 보내기 전 '설거지 위험' 검사 → 위험하면 알림 자체를 안 보냄
   - 최근 3일 증자·워런트·병합 뉴스        - 최근 3개월 '솟았다가 원위치' 이력 2회 이상
   - 다음 저항까지 +20% 미만 (먹을 자리 부족) - 당일 고점 대비 -15% 이상 밀림
   - VWAP 이탈 · VWAP +30% 과열 · 최근 15분 매도 우위 · 고점 찍고 10분 넘게 정체
track()/watch(): 추천(알림) 종목을 이틀간 1분마다 추적 → 설거지 신호 나오면 🚨 알림
   - 추천 후 +10% 이상 갔다가 고점 대비 -15% 급락     - 손절 라인 이탈
   - 추천 후 증자·병합 뉴스 등장                     - 급락 + VWAP 이탈 (1분봉 확인)
"""

import html
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import catalyst

log = logging.getLogger("guard")

# ═════════════════════ 설정 ═════════════════════
MIN_UPSIDE = 20.0        # 다음 저항까지 최소 여유 (%)
FADE_BLOCK = 15.0        # 당일 고점 대비 이만큼 밀렸으면 제외
VWAP_HOT = 1.30          # VWAP 대비 +30% 이상이면 과열
STALL_MIN = 10           # 고점 찍고 n분 넘게 신고가 없으면 정체
TRACK_DAYS = 2           # 추천 후 추적 기간 (거래일 기준 대략)
DD_WARN = 15.0           # 추천 후 고점 대비 -15% → 설거지 의심
STOP_PCT = 8.0           # 기본 손절 (추천가 대비)
NEWS_EVERY = 600         # 추적 종목 뉴스 확인 간격 (초)
CACHE_SEC = 900          # 같은 종목 재검사 간격 (초)
DILUTE = ["offering", "registered direct", "private placement", "warrant", "at-the-market",
          "reverse split", "reverse stock split", "priced", "shelf", "dilution"]
# ═══════════════════════════════════════════════

_cache: dict = {}


def _today_bars(t: str) -> pd.DataFrame:
    try:
        b = yf.Ticker(t).history(period="1d", interval="1m", prepost=True).dropna(subset=["Close"])
        return b
    except Exception:
        return pd.DataFrame()


def dilution_news(t: str, hours: int = 72) -> str | None:
    now = datetime.now(timezone.utc)
    try:
        for it in (yf.Ticker(t).news or [])[:15]:
            c = it.get("content") or it
            title = c.get("title") or ""
            ts = None
            if c.get("pubDate"):
                ts = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            elif it.get("providerPublishTime"):
                ts = datetime.fromtimestamp(it["providerPublishTime"], timezone.utc)
            if ts and now - ts > timedelta(hours=hours):
                continue
            if any(w in title.lower() for w in DILUTE):
                return title
    except Exception:
        pass
    return None


def pre_check(t: str, price: float, chg: float | None = None) -> tuple[bool, list[str], dict]:
    """→ (통과 여부, 위험 사유, {"upside": 다음 저항까지 %, None=신고가 영역})"""
    hit = _cache.get(t)
    if hit and time.time() - hit[0] < CACHE_SEC and not hit[1][0]:
        return hit[1]                                   # 최근에 탈락한 종목은 15분간 재검사 생략
    bad, info, hard = [], {"upside": None}, False
    news = dilution_news(t)
    if news:
        bad.append(f"증자·병합 뉴스: {news[:50]}")
        hard = True
    try:
        d = yf.Ticker(t).history(period="6mo", interval="1d").dropna(subset=["Close"])
        today = datetime.now(catalyst.ETZ).date()
        d = d[[ts.date() < today for ts in d.index]]
        if len(d) > 10:
            r = d.iloc[-63:]
            pumps = int(((r["High"] / r["Open"] - 1 >= 0.5) & (r["Close"] / r["Open"] - 1 <= 0.1)).sum())
            if pumps >= 2:
                bad.append(f"설거지 이력 {pumps}회(3개월)")
                hard = True
            above = d["High"][d["High"] > price * 1.05]
            if len(above):
                up = (float(above.min()) / price - 1) * 100
                info["upside"] = round(up)
                info["resist"] = float(above.min())
                if up < MIN_UPSIDE:
                    bad.append(f"저항 +{up:.0f}% (먹을 자리 부족)")
                    hard = True
    except Exception as e:
        log.debug("%s 일봉 실패: %s", t, e)
    b = _today_bars(t)
    if len(b) >= 10:
        c, o, h, v = b["Close"], b["Open"], b["High"], b["Volume"]
        last = price or float(c.iloc[-1])
        hi = float(h.max())
        vwap = float(((h + b["Low"] + c) / 3 * v).sum() / max(float(v.sum()), 1))
        fade = (hi - last) / hi * 100 if hi else 0
        info["vwap"] = vwap
        if fade >= FADE_BLOCK:
            bad.append(f"당일 고점 대비 -{fade:.0f}%")
            hard = True
        if vwap and last < vwap * 0.98:
            bad.append("VWAP 아래")
            hard = True
        if vwap and last > vwap * VWAP_HOT:
            bad.append(f"VWAP +{(last / vwap - 1) * 100:.0f}% 과열")
            hard = True
        red = float(v.iloc[-15:][c.iloc[-15:] < o.iloc[-15:]].sum())
        green = float(v.iloc[-15:][c.iloc[-15:] > o.iloc[-15:]].sum())
        if red > green * 1.5 and red > 0:
            bad.append("최근 15분 매도 우위")
            hard = True
        since = len(b) - 1 - int(h.values.argmax())
        if since >= STALL_MIN and fade >= 5:
            bad.append(f"고점 후 {since}분 정체")
            hard = True
    res = (not hard, bad, info)
    _cache[t] = (time.time(), res)
    if hard:
        log.info("설거지 필터 제외 %s: %s", t, ", ".join(bad))
    return res


def upside_line(info: dict) -> str:
    if info.get("upside") is None:
        return "📈 상승 여력: 6개월 고점 위 (위에 매물 없음)"
    return f"📈 상승 여력: 다음 저항 {catalyst._p(info.get('resist'))} (+{info['upside']}%)"


# ───────────────── 추천 후 감시 ─────────────────
def track(state: dict, t: str, price: float, kind: str, stop: float | None = None) -> None:
    tr = state.setdefault("track", {})
    if t in tr or not price:
        return
    tr[t] = {"price": price, "peak": price, "kind": kind, "at": datetime.now(timezone.utc).isoformat(),
             "stop": stop or price * (1 - STOP_PCT / 100), "sent": [], "news_at": 0}


def watch(state: dict, send) -> None:
    tr = state.get("track") or {}
    now = datetime.now(timezone.utc)
    for t in [t for t, e in tr.items() if now - datetime.fromisoformat(e["at"]) > timedelta(days=TRACK_DAYS + 1)]:
        del tr[t]
    now_et = now.astimezone(catalyst.ETZ)
    if not tr or not catalyst.session_fields(now_et):
        return
    q = catalyst.quotes(list(tr))
    for t, e in list(tr.items()):
        qt = q.get(t)
        if not qt:
            continue
        price, chg, vol = catalyst.live(qt, now_et)
        if not price:
            continue
        e["peak"] = max(e["peak"], price)
        gain = (price / e["price"] - 1) * 100
        peak_g = (e["peak"] / e["price"] - 1) * 100
        dd = (1 - price / e["peak"]) * 100
        level, why = None, []
        if price <= e["stop"] and "stop" not in e["sent"]:
            level, why = "stop", [f"손절 라인 {catalyst._p(e['stop'])} 이탈"]
        elif peak_g >= 10 and dd >= DD_WARN and "dd" not in e["sent"]:
            level, why = "dd", [f"고점 대비 -{dd:.0f}% 급락"]
        elif time.time() - e.get("news_at", 0) > NEWS_EVERY and "dil" not in e["sent"]:
            e["news_at"] = time.time()
            n = dilution_news(t, hours=24)
            if n:
                level, why = "dil", [f"증자·병합 뉴스: {n[:60]}"]
        if not level:
            continue
        b = _today_bars(t)
        if len(b) >= 10:
            v, c = b["Volume"], b["Close"]
            vwap = float(((b["High"] + b["Low"] + c) / 3 * v).sum() / max(float(v.sum()), 1))
            if price < vwap:
                why.append(f"VWAP {catalyst._p(vwap)} 이탈")
            red = float(v.iloc[-15:][c.iloc[-15:] < b["Open"].iloc[-15:]].sum())
            green = float(v.iloc[-15:][c.iloc[-15:] > b["Open"].iloc[-15:]].sum())
            if red > green * 1.5 and red > 0:
                why.append("대량 매도 출회")
        e["sent"].append(level)
        title = "🛑 손절 신호" if level == "stop" else "🚨 설거지 의심"
        act = ("👉 보유 중이면 정리 고려 — 셋업 실패" if level == "stop" else
               "👉 보유 중이면 익절·비중 축소 고려 · 재진입은 VWAP 회복 확인 후")
        at = datetime.fromisoformat(e["at"]).astimezone(catalyst.KST).strftime("%m/%d %H:%M")
        send(f"{title} <b>{t}</b> ({e['kind']} 알림 {at} 이후)\n"
             f"💰 알림가 {catalyst._p(e['price'])} → 최고 {catalyst._p(e['peak'])} (+{peak_g:.0f}%) → 현재 {catalyst._p(price)} ({gain:+.0f}%)\n"
             f"⚠️ {' · '.join(html.escape(w) for w in why)}\n{act}")
        if level == "stop":
            del tr[t]


def track_text(state: dict) -> str:
    tr = state.get("track") or {}
    if not tr:
        return "👀 추적 중인 추천 종목이 없어요."
    rows = [f"{t:<5} {e['kind']:<4} {catalyst._p(e['price']):>8} 최고 +{(e['peak'] / e['price'] - 1) * 100:>3.0f}%"
            for t, e in tr.items()]
    return "👀 <b>추천 후 추적 중</b> (설거지 신호 나오면 🚨 알림)\n<pre>" + html.escape("\n".join(rows)) + "</pre>"
