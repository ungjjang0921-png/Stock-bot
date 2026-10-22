"""
매수 신호 → 가상 체결 → 매도 신호 (앵봇 형식)
════════════════════════════════════════════
🔔 매수 신호   : 신호가 · 지정가(-5%) · 원화 환산 · 근거
✅ 체결 (가상) : 지정가 닿으면 지정가 체결, 3분 안 안 닿으면 시장가 전환 (반등형은 5분 뒤 취소)
🎯 절반 익절   : +15% 도달
🏁 매도 신호   : 손절 · 고점 대비 -10% 트레일링 · 증자 뉴스 · 장 마감 정리
/signals       : 가상 매매 성적 (승률 · 평균 수익)
"""

import html
import logging
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime

import yfinance as yf

import catalyst
import guard

log = logging.getLogger("signals")

# ═════════════════════ 설정 ═════════════════════
LIMIT_PCT = 5.0          # 지정가 = 신호가 -5%
FILL_WAIT = 180          # 3분 안 안 닿으면 시장가
REBOUND_WAIT = 300       # 반등형은 5분 안 안 닿으면 취소
STOP_PCT = 10.0          # 기본 손절
STOP_DILUTE = 30.0       # 최근 증자 공시 종목 손절 폭 (변동성 큼)
TP1_PCT = 15.0           # 절반 익절
TRAIL_ARM = 10.0         # +10% 넘으면 트레일링 시작
TRAIL_PCT = 10.0         # 고점 대비 -10%면 매도
MAX_OPEN = 8             # 동시 보유 한도
ICON = {"급등": "🚀", "출발": "🛫", "반등": "🔄", "촉매": "🔥", "점화": "⚡"}
SUFFIX = [" Inc.", " Inc", " Corp.", " Corp", " Corporation", " Holdings", " Ltd.", " Ltd", " plc", " Co.",
          " Class A", " Common Stock", " Ordinary Shares", ",", " N.V.", " S.A.", " Limited", " Group"]
TRANSLATE = None         # stock_alert_bot 이 번역 함수를 넣어줌
# ═══════════════════════════════════════════════

_fx = {"t": 0, "v": None}


def krw_rate() -> float | None:
    if time.time() - _fx["t"] > 3600 or not _fx["v"]:
        try:
            v = yf.Ticker("KRW=X").history(period="5d", interval="1d")["Close"].dropna()
            if len(v):
                _fx["v"] = float(v.iloc[-1])
        except Exception as e:
            log.warning("환율 조회 실패: %s", e)
        _fx["t"] = time.time()
    return _fx["v"]


def won(p: float) -> str:
    r = krw_rate()
    return f"{catalyst._p(p)}({p * r:,.0f}원)" if r and p else catalyst._p(p)


def kname(state: dict, t: str, name: str) -> str:
    cache = state.setdefault("kname", {})
    if t in cache:
        return cache[t]
    n = name or ""
    for s in SUFFIX:
        n = n.replace(s, "")
    n = n.strip()
    ko = n
    if TRANSLATE and n:
        try:
            ko = TRANSLATE(n).replace("주식회사", "").replace("(주)", "").strip() or n
        except Exception:
            pass
    cache[t] = ko
    if len(cache) > 3000:
        cache.pop(next(iter(cache)))
    return ko


def _head(kind: str, t: str, nm: str) -> str:
    return f"{ICON.get(kind, '🔔')} {kind}\n<b>{t}</b> · {html.escape(nm)}"


def buy(state: dict, send, t: str, price: float, kind: str, why: list[str], name: str = "",
        dilute: bool | None = None) -> bool:
    """매수 신호 발송. 같은 종목은 하루 한 번."""
    if not price:
        return False
    today = datetime.now(catalyst.ETZ).date().isoformat()
    sent = state.setdefault("sig_today", {})
    for d in [d for d in sent if d != today]:
        del sent[d]
    if t in sent.setdefault(today, []):
        return False
    pos = state.setdefault("pos", {})
    if t in pos or len([p for p in pos.values() if p["st"] == "open"]) >= MAX_OPEN:
        return False
    if dilute is None:
        dilute = bool(guard.dilution_news(t, hours=14 * 24))
    sent[today].append(t)
    limit = price * (1 - LIMIT_PCT / 100)
    if not name:
        name = (catalyst.quotes([t]).get(t) or {}).get("description") or ""
    nm = kname(state, t, name)
    wait = "5분 안 체결 안 되면 취소" if kind == "반등" else "3분 안 체결 안 되면 시장가"
    lines = [f"🔔 <b>매수 신호</b> · {_head(kind, t, nm)}",
             f"신호가  {won(price)}", f"지정가  {won(limit)}  (-{LIMIT_PCT:.0f}%)", wait,
             "매도는 🏁 알림을 따라 주세요"]
    if why:
        lines.append("💡 " + html.escape(" · ".join(w for w in why if w)))
    if dilute:
        lines.append(f"📄 최근 14일 증자·공모 공시 종목 — 손절 -{STOP_DILUTE:.0f}%")
    send("\n".join(lines))
    pos[t] = {"st": "pending", "kind": kind, "name": nm, "sig": price, "limit": limit,
              "at": time.time(), "stop_pct": STOP_DILUTE if dilute else STOP_PCT,
              "sess": (catalyst.session_fields(datetime.now(catalyst.ETZ)) or (0, 0, 0, ""))[3]}
    catalyst.log_alert(state, "매수신호", t, price, sig_kind=kind)
    return True


def _sell(state, send, t, p, price, reason):
    ret = (price / p["fill"] - 1) * 100
    if p.get("half"):                                           # 절반은 +15%에 익절
        ret = (ret + (p["half"] / p["fill"] - 1) * 100) / 2
    held = int((time.time() - p["fill_at"]) // 60)
    held_s = f"{held // 60}시간 {held % 60}분" if held >= 60 else f"{held}분"
    icon = "🟢" if ret > 0 else "🔴"
    send(f"🏁 <b>매도 신호</b> · {_head(p['kind'], t, p['name'])}\n"
         f"매도가  {won(price)}\n"
         f"{icon} 체결가 {catalyst._p(p['fill'])} 대비 {ret:+.1f}% · 보유 {held_s}\n"
         f"사유: {html.escape(reason)}")
    hist = state.setdefault("sig_hist", [])
    hist.append({"t": t, "kind": p["kind"], "ret": round(ret, 2), "at": datetime.now(timezone.utc).isoformat(),
                 "reason": reason})
    state["sig_hist"] = hist[-1000:]
    del state["pos"][t]


def manage(state: dict, send) -> None:
    pos = state.get("pos") or {}
    if not pos:
        return
    now_et = datetime.now(catalyst.ETZ)
    f = catalyst.session_fields(now_et)
    if not f:
        return
    q = catalyst.quotes(list(pos))
    now = time.time()
    for t, p in list(pos.items()):
        qt = q.get(t)
        if not qt:
            continue
        price, chg, vol = catalyst.live(qt, now_et)
        if not price:
            continue
        el = now - p["at"]
        el_s = f"{int(el // 60)}분 {int(el % 60)}초" if el >= 60 else f"{int(el)}초"
        if p["st"] == "pending":
            fill, how = None, ""
            if price <= p["limit"]:
                fill, how = p["limit"], "지정가 체결"
            elif p["kind"] == "반등" and el >= REBOUND_WAIT:
                send(f"⏹ <b>추가 매수 취소 (가상)</b> · {_head(p['kind'], t, p['name'])}\n5분 안 체결 안 됨")
                del pos[t]
                continue
            elif p["kind"] != "반등" and el >= FILL_WAIT:
                fill, how = price, "시장가 전환"
            if fill:
                p.update(st="open", fill=fill, fill_at=now, peak=fill, stop=fill * (1 - p["stop_pct"] / 100))
                send(f"✅ <b>체결 (가상)</b> · {_head(p['kind'], t, p['name'])}\n"
                     f"체결가  {won(fill)}\n"
                     f"신호가 대비 {(fill / p['sig'] - 1) * 100:+.1f}% · 신호 뒤 {el_s} · {how}\n"
                     f"🛑 손절 {catalyst._p(p['stop'])} (-{p['stop_pct']:.0f}%)")
            continue
        # ── 보유 중 ──
        p["peak"] = max(p["peak"], price)
        gain = (price / p["fill"] - 1) * 100
        peak_g = (p["peak"] / p["fill"] - 1) * 100
        if price <= p["stop"]:
            _sell(state, send, t, p, price, f"손절 라인 {catalyst._p(p['stop'])} 이탈")
            continue
        if not p.get("half") and gain >= TP1_PCT:
            p["half"] = price
            p["stop"] = max(p["stop"], p["fill"])                  # 절반 익절 후 손절을 본전으로
            send(f"🎯 <b>절반 익절</b> · {_head(p['kind'], t, p['name'])}\n"
                 f"익절가  {won(price)} (+{gain:.1f}%)\n나머지는 본전 손절 · 🏁 알림까지 보유")
            continue
        if peak_g >= TRAIL_ARM and price <= p["peak"] * (1 - TRAIL_PCT / 100):
            _sell(state, send, t, p, price, f"고점 {catalyst._p(p['peak'])} 대비 -{TRAIL_PCT:.0f}% (설거지 방어)")
            continue
        if now - p.get("news_at", 0) > 600:
            p["news_at"] = now
            n = guard.dilution_news(t, hours=24)
            if n and (now - p["fill_at"]) > 60 and not p.get("news_seen"):
                p["news_seen"] = True
                if "fill_news" not in p:
                    p["fill_news"] = n
                if gain > 0 or price < p["fill"]:
                    _sell(state, send, t, p, price, f"보유 중 증자·희석 뉴스: {n[:50]}")
                    continue
        t_et = now_et.time()
        if p["sess"] in ("프리장", "정규장") and dtime(15, 50) <= t_et < dtime(16, 0):
            _sell(state, send, t, p, price, "장 마감 전 정리")
        elif p["sess"] == "애프터장" and t_et >= dtime(19, 50):
            _sell(state, send, t, p, price, "애프터장 마감 전 정리")


def stats_text(state: dict) -> str:
    h = state.get("sig_hist") or []
    pos = state.get("pos") or {}
    if not h:
        body = "아직 끝난 가상 매매가 없어요."
    else:
        rows = []
        for k in ["전체"] + sorted({x["kind"] for x in h}):
            xs = [x for x in h if k == "전체" or x["kind"] == k]
            win = sum(x["ret"] > 0 for x in xs) / len(xs) * 100
            avg = sum(x["ret"] for x in xs) / len(xs)
            rows.append(f"{k:<4} {len(xs):>4}건 승률 {win:>3.0f}% 평균 {avg:+5.1f}% 최고 {max(x['ret'] for x in xs):+.0f}%")
        body = "<pre>" + html.escape("\n".join(rows)) + "</pre>"
    open_ = [f"{t} {p['kind']} {'대기' if p['st'] == 'pending' else catalyst._p(p['fill'])}" for t, p in pos.items()]
    return (f"💹 <b>가상 매매 성적</b> (신호대로 사고팔았다면)\n{body}"
            + (f"\n보유·대기: {', '.join(open_)}" if open_ else ""))
