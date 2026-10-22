"""
종목 상세 리포트 데이터 (docs/r/TICKER.json → docs/r.html 이 그려줌)
════════════════════════════════════════════
1. 회사       : 가격·시총·60일 차트·사업 요약(번역)·매출/손익·발행/유통주식·거래량 폭발 분석·강도
2. 진입 판단  : 가격 사다리(baby shelf 탈출가·$1·MVLS·물량 출회가·저항·지지) · 가산/감점 · 과거 조달 패턴
2-1. 희석 물량: 워런트·옵션·RSU·전환사채·우선주 (SEC XBRL) · 상단 돌파 가능성
3. 런웨이     4. 목적(얼마까지·언제까지·어떻게 + 종합 판단)   5. 예정 이벤트
6. 보유자·내부자 (Yahoo 기관·내부자 거래)   7. 메모(브라우저 저장)   8. 타임라인
"""

import json
import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("report")

BABY_SHELF = 75e6        # 공모 유통시총 7,500만$ 미만 → S-3 연간 1/3 한도
MVLS_MIN = 35e6          # 나스닥 캐피털마켓 상장시총 기준
DILUTION_FORMS = ("424B4", "424B5")


def _n(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def P(v):
    return "-" if v is None else (f"${v:.4f}" if v < 1 else f"${v:.2f}")


def pct(a, b):
    return None if not a or not b else round((a / b - 1) * 100)


def big(v):
    if v is None:
        return "-"
    v = float(v)
    return f"{v / 1e9:.2f}B" if v >= 1e9 else f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


# ───────────────── SEC XBRL ─────────────────
def fact(cf: dict, names: list[str], units=("USD", "shares", "USD/shares")) -> dict | None:
    facts = (cf or {}).get("facts", {})
    for n in names:
        f = facts.get("us-gaap", {}).get(n) or facts.get("dei", {}).get(n)
        if not f:
            continue
        for u in units:
            pts = [p for p in f.get("units", {}).get(u, []) if "end" in p]
            if pts:
                p = max(pts, key=lambda p: (p["end"], p.get("filed", "")))
                return {"val": p["val"], "end": p["end"], "start": p.get("start"), "form": p.get("form", "")}
    return None


def period_note(f):
    if not f or not f.get("start"):
        return ""
    m = round((date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days / 30.4)
    return f"{m}개월({f['end'][:7]})"


def dilutive(cf: dict, price: float) -> dict:
    rows = []

    def add(kind, holder, qty, px, note):
        if not qty:
            return
        st = "출회 가능" if (px is None or px <= price) else "근접" if px <= price * 1.5 else "OTM"
        rows.append({"kind": kind, "holder": holder, "qty": qty, "px": px, "px_pct": pct(px, price) if px else None,
                     "st": st, "note": note})

    w = fact(cf, ["ClassOfWarrantOrRightOutstanding"], ("shares",))
    wp = fact(cf, ["ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
                   "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights"], ("USD/shares",))
    if w:
        add("워런트", "투자자", w["val"], wp["val"] if wp else None, f"{w['end']} 기준 (여러 종류면 합계·대표 행사가)")
    o = fact(cf, ["ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber"], ("shares",))
    op = fact(cf, ["ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice"],
              ("USD/shares",))
    if o:
        add("옵션", "임원·직원", o["val"], op["val"] if op else None, f"{o['end']} 기준 · 가중평균 행사가")
    r = fact(cf, ["ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber"],
             ("shares",))
    if r:
        add("RSU", "임원·직원", r["val"], 0.0, f"{r['end']} 기준 · 미확정분")
    pf = fact(cf, ["PreferredStockSharesOutstanding"], ("shares",))
    if pf and pf["val"]:
        add("우선주", "투자자", pf["val"], None, f"{pf['end']} 기준 · 전환 조건은 원문 확인")
    cv = fact(cf, ["ConvertibleNotesPayableCurrent", "ConvertibleNotesPayable", "ConvertibleDebtCurrent",
                   "ConvertibleDebtNoncurrent"], ("USD",))
    conv = cv["val"] if cv and cv["val"] else None
    total = sum(x["qty"] for x in rows)
    now = sum(x["qty"] for x in rows if x["st"] == "출회 가능")
    return {"rows": rows, "total": total, "now": now, "conv_usd": conv}


# ───────────────── 가격·거래량 ─────────────────
def spike(df: pd.DataFrame, fl: float | None, fs: list[dict]) -> dict | None:
    if df is None or len(df) < 30:
        return None
    v, c, h, l, o = df["Volume"], df["Close"], df["High"], df["Low"], df["Open"]
    rel = v / v.rolling(20).mean().shift(1)
    r20 = rel.iloc[-20:]
    if r20.max() != r20.max() or r20.max() < 5:
        return None
    i = df.index.get_loc(r20.idxmax())
    d = df.index[i].date()
    chg = (c.iloc[i] / c.iloc[i - 1] - 1) * 100
    fade = (1 - c.iloc[i] / h.iloc[i]) * 100
    turn = v.iloc[i] / fl if fl else None
    turn7 = v.iloc[i:i + 7].sum() / fl if fl else None
    same = [f for f in fs if f["date"] == d.isoformat()]
    after_low = float(l.iloc[i:].min())
    held = after_low >= l.iloc[i] * 0.98
    if turn and turn >= 0.8 and (fade >= 5 or chg < 0):
        verdict = "분배 의심 — 던질 물량이 손바뀜으로 나갔을 수 있음"
    elif -5 <= chg <= 15 and fade < 5:
        verdict = "매집 의심 — 거래량 대비 가격 변화 작음"
    elif chg >= 30:
        verdict = "급등형 — 재료 확인 필요"
    else:
        verdict = "중립"
    return {"date": d.isoformat(), "vol": float(v.iloc[i]), "x": round(float(rel.iloc[i])),
            "avg20": float(v.iloc[i - 20:i].mean()), "chg": round(chg, 1), "fade": round(fade, 1),
            "turn": None if turn is None else round(turn, 1), "turn7": None if turn7 is None else round(turn7, 1),
            "filing": ", ".join(f["form"] for f in same) or "없음", "low": float(l.iloc[i]), "held": held,
            "verdict": verdict}


def resist(df, price):
    if df is None:
        return None
    above = df["High"].iloc[-126:]
    above = above[above > price * 1.05]
    return float(above.min()) if len(above) else None


# ───────────────── 과거 조달 패턴 ─────────────────
def past_raises(fs: list[dict], df: pd.DataFrame | None) -> list[dict]:
    out = []
    cutoff = (date.today() - timedelta(days=730)).isoformat()
    pr_dates = [f["date"] for f in fs if f["form"].startswith("8-K") and any(x in f.get("items", "") for x in ("7.01", "8.01"))]
    for f in fs:
        if f["date"] < cutoff:
            continue
        is_raise = f["form"] in DILUTION_FORMS or (f["form"].startswith("8-K") and "3.02" in f.get("items", ""))
        if not is_raise:
            continue
        d = date.fromisoformat(f["date"])
        pr = [x for x in pr_dates if 0 < (d - date.fromisoformat(x)).days <= 14]
        px = run = None
        if df is not None and len(df):
            idx = [ts.date() for ts in df.index]
            j = next((k for k, x in enumerate(idx) if x >= d), None)
            if j is not None and j >= 20:
                px = float(df["Close"].iloc[j])
                run = round((float(df["High"].iloc[j - 10:j + 1].max()) / float(df["Close"].iloc[j - 20]) - 1) * 100)
        out.append({"date": f["date"], "form": f["form"], "px": px, "run": run, "pr": pr[-1] if pr else None,
                    "url": f["url"]})
    return out[:8]


# ───────────────── 메인 ─────────────────
def build(it: dict, df: pd.DataFrame | None, fs: list[dict], cf: dict, prof: dict, cal: list[dict],
          classify, ai_text: str = "") -> dict:
    price = it["price"]
    sh = it.get("sh") or {}
    shares = sh.get("now")
    fl = it.get("float")
    today = date.today()

    # 1. 회사
    spark = [round(float(x), 6) for x in df["Close"].iloc[-60:]] if df is not None else []
    rev = fact(cf, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], ("USD",))
    ni = fact(cf, ["NetIncomeLoss"], ("USD",))
    sp = spike(df, fl, fs)
    insider_buys = sum(1 for x in prof.get("insiders", []) if "매수" in x["act"])
    signals = []
    if sp:
        signals.append(f"거래량 폭발 ×{sp['x']}")
        if "분배" in sp["verdict"]:
            signals.append("손바뀜 분배 의심")
        elif "매집" in sp["verdict"]:
            signals.append("매집 의심")
    if insider_buys:
        signals.append(f"내부자 매수 {insider_buys}")
    if prof.get("inst"):
        signals.append(f"기관·펀드 {len(prof['inst'])}")
    tags = [{"name": t["name"], "level": t["level"]} for t in it.get("tags", [])]

    # 2. 가격 사다리
    lv = []
    hi60 = float(df["Close"].iloc[-60:].max()) if df is not None and len(df) else price
    if fl:
        pub = hi60 * fl
        if pub < BABY_SHELF:
            esc = BABY_SHELF / fl
            lv.append({"px": esc, "label": "baby shelf 탈출가 — 공모 유통시총 $75M 회복 · 선반 한도 해제",
                       "kind": "up", "tag": ""})
            baby = {"on": True, "cap": pub / 3, "esc": esc, "pub": pub}
        else:
            baby = {"on": False, "pub": pub}
    else:
        baby = None
    if price < 1:
        lv.append({"px": 1.0, "label": "상장 유지 최저호가 — $1 이상 10영업일", "kind": "up", "tag": "목표"})
    if shares and shares * price < MVLS_MIN:
        lv.append({"px": MVLS_MIN / shares, "label": "상장 유지 MVLS — 시총 $35M (나스닥 CM 기준)", "kind": "up", "tag": "목표"})
    dil = dilutive(cf, price)
    for r in dil["rows"]:
        if r["px"]:
            lv.append({"px": r["px"], "label": f"물량 출회 — {r['kind']} 행사가 ({big(r['qty'])}주)", "kind": "down",
                       "tag": "출회 가능" if r["px"] <= price else "근접" if r["px"] <= price * 1.5 else ""})
    rs = resist(df, price)
    if rs:
        lv.append({"px": rs, "label": "최근 6개월 저항(고점 매물)", "kind": "res", "tag": ""})
    support = sp["low"] if sp else (float(df["Low"].iloc[-20:].min()) if df is not None else None)
    if support:
        lv.append({"px": support, "label": "지지선 — " + (f"{sp['date']} 폭발일 저가" if sp else "20일 저가"),
                   "kind": "sup", "tag": ""})
    lv.append({"px": price, "label": "현재가", "kind": "now", "tag": ""})
    for x in lv:
        x["pct"] = pct(x["px"], price)
    lv.sort(key=lambda x: -x["px"])

    # 가산·감점
    score = []
    above = [x for x in lv if x["px"] > price * 1.02 and x["kind"] in ("up", "res")]
    if above:
        n = min(above, key=lambda x: x["px"])
        score.append({"pt": 2 if n["pct"] >= 50 else 1, "t": f"바로 위 목표까지 +{n['pct']}% ({P(n['px'])} {n['label'].split(' —')[0]})"})
    if sp and "분배" in sp["verdict"]:
        score.append({"pt": -2, "t": "손바뀜 분배 의심"})
    if dil["now"] and shares and dil["now"] / shares >= 0.1:
        score.append({"pt": -1, "t": f"지금 출회 가능 물량 {big(dil['now'])}주 (발행의 {dil['now'] / shares * 100:.0f}%)"})
    rw = it.get("runway")
    if rw is not None and rw < 3:
        score.append({"pt": -1, "t": f"런웨이 {rw:.1f}개월 — 다음 조달이 할인 공모일 가능성"})
    if support:
        score.append({"pt": 0, "t": f"지지선 {P(support)} 위 {pct(price, support)}%"})
    if insider_buys:
        score.append({"pt": 1, "t": f"내부자 매수 {insider_buys}건"})

    raises = past_raises(fs, df)

    # 2-1. 상단 돌파 가능성
    target = min(above, key=lambda x: x["px"]) if above else None
    overhead = sum(r["qty"] for r in dil["rows"] if r["px"] is not None and target and price < r["px"] <= target["px"])
    absorb = (overhead / sp["vol"]) if sp and sp["vol"] else None
    if target:
        brk = ("가능" if absorb is not None and absorb <= 0.5 else "부담" if absorb is not None and absorb > 1.5 else "중간")
    else:
        brk = "가능"
    breakout = {"target": target, "overhead": overhead, "absorb": None if absorb is None else round(absorb, 2),
                "verdict": brk, "range20": round((float(df["High"].iloc[-20:].max()) / float(df["Low"].iloc[-20:].min()) - 1) * 100)
                if df is not None else None}

    # 3. 런웨이
    burn = it.get("burn")
    cash_date = None
    runway = {"cash": it.get("cash"), "burn": burn, "months": rw, "equity": it.get("equity"), "empty": None}
    if rw is not None and rw < 99 and sh.get("now_date"):
        cash_date = date.fromisoformat(sh["now_date"])
        empty = cash_date + timedelta(days=int(rw * 30.4))
        runway["empty"] = empty.isoformat() + (" (이미 소진 추정)" if empty < today else "")

    # 4. 목적
    purposes = []
    tagmap = {t["name"]: t for t in it.get("tags", [])}
    dd = {d["label"]: d for d in it.get("ddays", [])}
    if "현금 조달" in tagmap or (rw is not None and rw < 12):
        tgt = target["px"] if target else price * 1.3
        purposes.append({"name": "현금 조달", "type": "물량형", "main": True, "upto": tgt, "upto_pct": pct(tgt, price),
                         "upto_note": "다음 저항·목표가 — 띄운 뒤 조달하는 경우가 많음",
                         "when": runway.get("empty"), "how": f"현금 {big(it.get('cash'))} · 월 소진 {big(burn)} · 런웨이 {rw:.1f}개월" if rw is not None else "재무 미확인",
                         "then": "조달 후 신규 투자자 물량(재판매 등록) 출회 주의"})
    if "상장 유지" in tagmap:
        d1 = next((d for k, d in dd.items() if "회복" in k or "제출" in k), None)
        purposes.append({"name": "상장 유지", "type": "기한형", "main": False, "upto": 1.0, "upto_pct": pct(1.0, price),
                         "upto_note": "$1 이상 10영업일", "when": d1["date"] if d1 else None, "dday": d1["d"] if d1 else None,
                         "how": tagmap["상장 유지"].get("note", ""), "then": "못 넘기면 역분할(병합) 또는 상폐 절차"})
    if "물량 출회" in tagmap or dil["now"]:
        purposes.append({"name": "물량 출회", "type": "물량형", "main": False,
                         "upto": min((r["px"] for r in dil["rows"] if r["px"]), default=None), "upto_pct": None,
                         "upto_note": "행사가 위에서 행사·매도 가능",
                         "when": None, "how": " · ".join(f"{r['kind']} {big(r['qty'])}주 @{P(r['px'])}" for r in dil["rows"][:3]),
                         "then": "가격이 행사가를 넘으면 매물로 나올 수 있음"})
    for p in purposes:
        if p.get("upto_pct") is None and p.get("upto"):
            p["upto_pct"] = pct(p["upto"], price)
    checks = []
    if price < 1 and not any(f["form"] == "PRE 14A" for f in fs[:30]):
        checks.append("역분할(병합) 승인을 위한 PRE 14A 제출 여부")
    if sp and sp["filing"] == "없음":
        checks.append(f"{sp['date']} 거래량 폭발 관련 Form 4·13D/G·8-K 제출 여부")
    if baby and baby.get("on"):
        checks.append("S-3 선반 남은 한도 (10-Q·424B5 원문)")

    # 5. 예정 이벤트
    events = [{"date": d["date"], "d": d["d"], "t": d["label"]} for d in it.get("ddays", [])]
    events += [{"date": e["date"], "d": e["d"], "t": e["ev"]} for e in cal if e["t"] == it["t"]]
    last_raise = next((f for f in fs if f["form"] in DILUTION_FORMS), None)
    if last_raise:
        roll = date.fromisoformat(last_raise["date"]) + timedelta(days=365)
        if roll >= today:
            events.append({"date": roll.isoformat(), "d": (roll - today).days, "t": "S-3 선반 1년 롤링 (한도 일부 회복)"})
    seen, ev2 = set(), []
    for e in sorted(events, key=lambda e: e["date"]):
        if (e["date"], e["t"]) not in seen and e["d"] >= -3:
            seen.add((e["date"], e["t"])); ev2.append(e)

    # 8. 타임라인
    tl = []
    for f in fs[:60]:
        side, lab = classify(f)
        if side == "중립" and f["form"] in ("10-Q", "10-K", "DEF 14A", "6-K", "S-8", "4", "3", "SC 13G/A"):
            continue
        tl.append({"date": f["date"], "kind": "공시", "side": side, "t": f"{f['form']} — {lab}", "url": f["url"]})
    if df is not None and len(df) > 25:
        rel = df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
        chg = df["Close"].pct_change() * 100
        for ts in rel.iloc[-90:][rel.iloc[-90:] >= 5].index:
            tl.append({"date": ts.date().isoformat(), "kind": "거래량", "side": "중립",
                       "t": f"거래량 폭발 ×{rel[ts]:.0f} · 종가 {chg[ts]:+.0f}%", "url": ""})
    tl.sort(key=lambda x: x["date"], reverse=True)

    # 종합 판단 (AI 없으면 규칙 기반 문장)
    if ai_text:
        verdict = ai_text
    else:
        parts = []
        if rw is not None and rw < 6:
            parts.append(f"현금이 {big(it.get('cash'))}로 런웨이 {rw:.1f}개월이라 조달이 임박한 구조예요.")
        if baby and baby.get("on"):
            parts.append(f"공모 유통시총 {big(baby['pub'])}으로 baby shelf 적용 — S-3 연간 한도 약 {big(baby['cap'])}. "
                         f"{P(baby['esc'])} 위에서 한도가 풀려요.")
        if raises:
            n_pr = sum(1 for r in raises if r["pr"])
            parts.append(f"최근 2년 조달 {len(raises)}회" + (f", 그중 {n_pr}회는 보도자료(8-K) 직후" if n_pr else "") +
                         f" — 마지막 {raises[0]['date']}.")
        if "상장 유지" in tagmap:
            parts.append("$1 미달 상태라 기한 전 부양성 호재나 병합 발표가 나올 수 있어요.")
        if sp:
            parts.append(f"{sp['date']} 거래량 ×{sp['x']} 폭발 (공시: {sp['filing']}) — {sp['verdict']}.")
        if dil["now"] and shares:
            parts.append(f"지금 출회 가능한 희석 물량 {big(dil['now'])}주(발행의 {dil['now'] / shares * 100:.0f}%).")
        verdict = " ".join(parts) or "뚜렷한 부양·조달 신호가 없어요."

    return {
        "t": it["t"], "name": it["name"], "exch": prof.get("exch", ""), "price": price, "chg": it.get("chg"),
        "mcap": it.get("mcap"), "spark": spark, "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tags": tags, "signals": signals, "strength": it.get("strength"),
        "summary": prof.get("summary_ko", ""), "sector": prof.get("sector", ""),
        "rev": rev, "rev_note": period_note(rev), "ni": ni, "ni_note": period_note(ni),
        "shares": shares, "shares_date": sh.get("now_date"), "float": fl, "growth": sh.get("growth"),
        "spike": sp, "levels": lv, "score": score, "baby": baby, "raises": raises,
        "dil": dil, "breakout": breakout, "runway": runway, "purposes": purposes, "checks": checks,
        "verdict": verdict, "events": ev2, "inst": prof.get("inst", []), "insiders": prof.get("insiders", []),
        "pct_insider": prof.get("pct_insider"), "pct_inst": prof.get("pct_inst"),
        "timeline": tl[:30], "tags_raw": [t.get("note", "") for t in it.get("tags", [])],
    }


def profile(t: str, translate) -> dict:
    """Yahoo: 사업 요약·거래소·기관 보유·내부자 거래 (하루 1번 캐시 권장)"""
    tk = yf.Ticker(t)
    out = {"inst": [], "insiders": []}
    try:
        info = tk.info or {}
        s = info.get("longBusinessSummary") or ""
        s = ". ".join(s.split(". ")[:3])[:600]
        out["summary_ko"] = translate(s) if s else ""
        out["exch"] = info.get("exchange", "")
        out["sector"] = " · ".join(x for x in (info.get("sector"), info.get("industry")) if x)
        out["pct_insider"] = _n(info.get("heldPercentInsiders"))
        out["pct_inst"] = _n(info.get("heldPercentInstitutions"))
    except Exception as e:
        log.debug("%s info 실패: %s", t, e)
    try:
        ih = tk.institutional_holders
        if ih is not None and len(ih):
            for _, r in ih.head(6).iterrows():
                out["inst"].append({"name": str(r.get("Holder", "")), "pct": _n(r.get("pctHeld")),
                                    "shares": _n(r.get("Shares")), "date": str(r.get("Date Reported", ""))[:10]})
    except Exception:
        pass
    try:
        tr = tk.insider_transactions
        if tr is not None and len(tr):
            for _, r in tr.head(10).iterrows():
                txt = str(r.get("Text", "") or r.get("Transaction", ""))
                low = txt.lower()
                act = "매수" if "purchase" in low or "buy" in low else "매도" if "sale" in low or "sell" in low \
                    else "옵션 행사" if "exercise" in low else "부여" if "grant" in low or "award" in low else "기타"
                out["insiders"].append({"name": str(r.get("Insider", "")), "pos": str(r.get("Position", "")),
                                        "act": act, "shares": _n(r.get("Shares")), "value": _n(r.get("Value")),
                                        "date": str(r.get("Start Date", ""))[:10], "text": txt[:80]})
    except Exception:
        pass
    return out
