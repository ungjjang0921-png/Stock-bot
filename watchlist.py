"""
펌핑 목적 워치리스트 + 통합점수 대시보드 생성기
════════════════════════════════════════════
1) 나스닥 저가 소형주 수집 (TradingView 스캐너)
2) SEC 공시·재무 분석 → '주가를 띄워야 할 이유' 태그
     상장유지 · 현금조달 · 물량출회 · 병합 · 매집의심 · 소형유통
3) 차트 진입 조건 n/5 + 같은 종목군 과거 유사 사례로 예상 고점 배수 추산
4) docs/index.html (GitHub Pages) 생성 + 텔레그램 요약 전송

환경변수
  TELEGRAM_TOKEN  : 텔레그램 봇 토큰
  SEC_EMAIL       : SEC 접속용 연락 이메일 (SEC 규정상 필수)
  PAGES_URL       : 대시보드 주소 (텔레그램 링크용)
  ANTHROPIC_API_KEY (선택): 있으면 상위 종목에 AI 종합판단 추가 (유료)
"""

import html
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ═════════════════════ 설정 ═════════════════════
EXCHANGES = ["NASDAQ"]
MIN_PRICE, MAX_PRICE = 0.05, 5.0
MIN_MCAP, MAX_MCAP = 1e6, 300e6
MIN_AVG_VOL = 100_000
MAX_UNIVERSE = 300
FORWARD_DAYS = 60                 # 유사 사례 고점 측정 기간 (거래일)
TOP_TELEGRAM = 10
AI_TOP = 10
AI_MODEL = "claude-sonnet-5-5"

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
SEC_EMAIL = os.getenv("SEC_EMAIL", "").strip()
PAGES_URL = os.getenv("PAGES_URL", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "index.html"
KST = ZoneInfo("Asia/Seoul")
# ═══════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("watchlist")
http = requests.Session()
UA_WEB = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36"}


def num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ───────────────── 1. 종목 수집 ─────────────────
def universe() -> list[dict]:
    cols = ["description", "close", "change", "market_cap_basic", "average_volume_10d_calc",
            "float_shares_outstanding", "volume"]
    body = {"markets": ["america"],
            "filter": [{"left": "exchange", "operation": "in_range", "right": EXCHANGES},
                       {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                       {"left": "close", "operation": "in_range", "right": [MIN_PRICE, MAX_PRICE]},
                       {"left": "market_cap_basic", "operation": "in_range", "right": [MIN_MCAP, MAX_MCAP]},
                       {"left": "average_volume_10d_calc", "operation": "greater", "right": MIN_AVG_VOL}],
            "columns": cols, "sort": {"sortBy": "average_volume_10d_calc", "sortOrder": "desc"},
            "range": [0, MAX_UNIVERSE]}
    for attempt in range(3):
        try:
            data = http.post("https://scanner.tradingview.com/america/scan", json=body,
                             headers=UA_WEB, timeout=30).json()
            out = []
            for row in data["data"]:
                d = dict(zip(cols, row["d"]))
                out.append({"t": row["s"].split(":")[-1].replace(".", "-"), "name": d["description"] or "",
                            "price": num(d["close"]), "chg": num(d["change"]),
                            "mcap": num(d["market_cap_basic"]), "float": num(d["float_shares_outstanding"]),
                            "avgvol": num(d["average_volume_10d_calc"])})
            log.info("대상 종목 %d개", len(out))
            return out
        except Exception as e:
            log.warning("종목 수집 실패 (%d/3): %s", attempt + 1, e)
            if "float_shares_outstanding" in cols and attempt == 0:
                pass
            time.sleep(3)
    return []


def daily(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            df = yf.download(chunk, period="1y", interval="1d", group_by="ticker", actions=True,
                             auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            log.warning("일봉 실패: %s", e)
            continue
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(subset=["Close"])
                if len(sub) >= 30:
                    out[t] = sub
            except KeyError:
                pass
        time.sleep(1)
    return out


# ───────────────── 2. SEC 공시·재무 ─────────────────
SEC_UA = {"User-Agent": f"stock-watchlist {SEC_EMAIL or 'contact@example.com'}"}
_last_sec = 0.0


def sec_get(url: str) -> dict | None:
    global _last_sec
    wait = 0.12 - (time.time() - _last_sec)          # SEC 규정: 초당 10회 이하
    if wait > 0:
        time.sleep(wait)
    _last_sec = time.time()
    try:
        r = http.get(url, headers=SEC_UA, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code != 404:
            log.warning("SEC %s → %s", url[-60:], r.status_code)
    except Exception as e:
        log.warning("SEC 요청 실패: %s", e)
    return None


def cik_map() -> dict[str, int]:
    data = sec_get("https://www.sec.gov/files/company_tickers.json") or {}
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


FORM_KO = {
    "8-K:3.01": "상장규정 미달 통지", "8-K:5.03": "정관 변경(병합 가능)", "8-K:3.02": "사모·미등록 발행",
    "8-K:1.01": "주요 계약 체결", "8-K:2.02": "실적 발표", "8-K:5.07": "주총 결과",
    "424B5": "공모·ATM 설명서", "424B4": "공모가 확정", "424B3": "재판매 설명서",
    "S-1": "증권신고서", "S-1/A": "증권신고서 정정", "S-3": "일괄신고(Shelf)", "S-3/A": "Shelf 정정",
    "EFFECT": "신고서 효력 발생", "DEF 14A": "주총 위임장", "PRE 14A": "주총 위임장(예비)",
    "NT 10-Q": "분기보고 지연", "NT 10-K": "연간보고 지연", "10-Q": "분기보고서", "10-K": "연간보고서",
    "S-8": "임직원 주식 등록",
}


def filings(cik: int) -> list[dict]:
    data = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if not data:
        return []
    r = data.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(r.get("form", [])[:300]):
        try:
            out.append({"form": form, "date": r["filingDate"][i], "items": r.get("items", [""] * 999)[i] or "",
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                               f"{r['accessionNumber'][i].replace('-', '')}/{r['primaryDocument'][i]}",
                        "period": r.get("reportDate", [""] * 999)[i] or ""})
        except (IndexError, KeyError):
            continue
    return out


def concept(cik: int, name: str) -> list[dict]:
    d = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{name}.json")
    return (d or {}).get("units", {}).get("USD", [])


def latest_value(facts: list[dict]) -> tuple[float, str] | None:
    pts = [f for f in facts if f.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A") and "end" in f]
    if not pts:
        return None
    f = max(pts, key=lambda f: (f["end"], f.get("filed", "")))
    return float(f["val"]), f["end"]


def monthly_burn(facts: list[dict]) -> float | None:
    """영업현금흐름 → 월 소진액 (+면 소진). 가장 최근 기간 중 가장 긴 기간 사용"""
    pts = [f for f in facts if "start" in f and "end" in f and f.get("form", "").startswith(("10-Q", "10-K"))]
    if not pts:
        return None
    end = max(f["end"] for f in pts)
    cand = [f for f in pts if f["end"] == end]
    f = max(cand, key=lambda f: (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days)
    months = max(((date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days) / 30.4, 1)
    return -float(f["val"]) / months


def financials(cik: int) -> dict:
    cash = None
    for name in ("CashAndCashEquivalentsAtCarryingValue",
                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"):
        cash = latest_value(concept(cik, name))
        if cash:
            break
    burn = monthly_burn(concept(cik, "NetCashProvidedByUsedInOperatingActivities"))
    eq = latest_value(concept(cik, "StockholdersEquity"))
    runway = None
    if cash and burn is not None:
        runway = 99.0 if burn <= 0 else cash[0] / burn
    return {"cash": cash[0] if cash else None, "cash_date": cash[1] if cash else None,
            "burn": burn, "runway": runway, "equity": eq[0] if eq else None}


# ───────────────── 3. 동기 태그 & 강도 ─────────────────
def days_ago(d: str, today: date) -> int:
    try:
        return (today - date.fromisoformat(d)).days
    except ValueError:
        return 9999


def has(fs, today, forms, within, item=None):
    for f in fs:
        if f["form"] in forms and days_ago(f["date"], today) <= within and (item is None or item in f["items"]):
            return f
    return None


def motive_tags(price: float, fl: float | None, fs: list[dict], fin: dict, df: pd.DataFrame | None,
                today: date) -> tuple[list[dict], list[dict], int]:
    """→ (동기 태그, D-day 목록, 원점수)"""
    tags, ddays, raw = [], [], 0

    def tag(name, level, note=""):
        tags.append({"name": name, "level": level, "note": note})

    # 상장유지
    n301 = has(fs, today, ["8-K", "8-K/A"], 365, "3.01")
    if n301 and price < 1:
        dl = date.fromisoformat(n301["date"]) + timedelta(days=180)
        tag("상장 유지", "기한형", "$1 미달 통지")
        ddays.append({"label": "$1 회복 기한(추정)", "date": dl.isoformat(), "d": (dl - today).days})
        raw += 2
    elif n301:
        dl = date.fromisoformat(n301["date"]) + timedelta(days=45)
        tag("상장 유지", "필요", "규정 미달 통지")
        ddays.append({"label": "회복 계획 제출(추정)", "date": dl.isoformat(), "d": (dl - today).days})
        raw += 2
    elif price < 1:
        tag("상장 유지", "가능", "$1 미만")
        raw += 1
    if fin.get("equity") is not None and fin["equity"] < 2.5e6:
        tag("자기자본", "필요", f"${fin['equity'] / 1e6:.1f}M < $2.5M")
        raw += 1

    # 현금조달
    rw = fin.get("runway")
    if rw is not None and rw < 6:
        tag("현금 조달", "급함", f"런웨이 {rw:.1f}개월"); raw += 2
    elif rw is not None and rw < 12:
        tag("현금 조달", "필요", f"런웨이 {rw:.1f}개월"); raw += 1
    atm = has(fs, today, ["424B5"], 120)
    shelf = has(fs, today, ["S-3", "S-3/A"], 3 * 365)
    s1 = has(fs, today, ["S-1", "S-1/A"], 120)
    if s1:
        tag("공모 준비", "진행", f"S-1 {s1['date']}"); raw += 1
    if atm:
        tag("ATM·공모", "활성", f"424B5 {atm['date']}")
    elif shelf:
        tag("Shelf", "보유", f"S-3 {shelf['date']}")

    # 물량출회
    resale = has(fs, today, ["424B3"], 180) or has(fs, today, ["8-K"], 180, "3.02")
    if resale:
        tag("물량 출회", "예정", f"{resale['form']} {resale['date']}"); raw += 1

    # 병합
    proxy = has(fs, today, ["PRE 14A", "DEF 14A"], 180)
    if proxy and price < 1:
        tag("병합", "안건 가능", f"{proxy['form']} {proxy['date']}"); raw += 1
    rs = 0
    if df is not None and "Stock Splits" in df:
        sp = df["Stock Splits"]
        rs = int(((sp > 0) & (sp < 1)).sum())
    if rs:
        tag("병합 이력", f"{rs}회", "최근 1년")

    # 지연·위험
    if has(fs, today, ["NT 10-Q", "NT 10-K"], 120):
        tag("공시 지연", "위험")

    # 소형 유통
    if fl and fl < 10e6:
        tag("소형 유통", f"{fl / 1e6:.1f}M"); raw += 1

    # 매집 의심: 최근 10일 중 거래량 10배+ 터졌는데 가격은 크게 안 오름
    if df is not None and len(df) > 30:
        v, c = df["Volume"], df["Close"]
        avg = v.rolling(20).mean().shift(1)
        for i in range(len(df) - 10, len(df)):
            if avg.iloc[i] and v.iloc[i] / avg.iloc[i] >= 10:
                move = (c.iloc[i] / c.iloc[i - 1] - 1) * 100
                if move < 40:
                    tag("매집 의심", f"x{v.iloc[i] / avg.iloc[i]:.0f}",
                        f"{df.index[i]:%m-%d} 거래량 폭증 · 종가 {move:+.0f}%")
                    raw += 1
                    break

    # 다음 실적 공시 마감 (분기말 +45일 추정)
    q = has(fs, today, ["10-Q", "10-K"], 200)
    if q and q.get("period"):
        nxt = date.fromisoformat(q["period"]) + timedelta(days=91 + 45)
        ddays.append({"label": "다음 분기보고 마감(추정)", "date": nxt.isoformat(), "d": (nxt - today).days})

    ddays = [d for d in ddays if d["d"] >= -7]
    ddays.sort(key=lambda d: d["d"])
    return tags, ddays, raw


def strength(raw: int) -> int:
    return 1 if raw <= 0 else 2 if raw <= 2 else 3 if raw <= 4 else 4 if raw <= 6 else 5


# ───────────────── 4. 차트 진입 조건 & 유사 사례 ─────────────────
def tech_frame(df: pd.DataFrame) -> pd.DataFrame:
    """날짜별 5가지 진입 조건 (각 1/0)"""
    c, l, v = df["Close"], df["Low"], df["Volume"]
    ma20 = c.rolling(20).mean()
    gap = (c / ma20 - 1) * 100
    up = c.diff() > 0
    upv = (v * up).rolling(10).sum()
    dnv = (v * (~up)).rolling(10).sum()
    delta = c.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    f = pd.DataFrame(index=df.index)
    f["ma20"] = (gap >= 0) & (gap <= 15)
    f["lows"] = l.rolling(3).min() > l.shift(3).rolling(3).min()
    f["upvol"] = upv > dnv * 1.2
    f["liq"] = (c * v).rolling(20).mean() >= 300_000
    f["cool"] = ((c / c.shift(5) - 1) < 0.5) & (rsi.fillna(50) < 75)
    f["stage"] = f[["ma20", "lows", "upvol", "liq", "cool"]].sum(axis=1)
    f["fwd"] = df["High"][::-1].rolling(FORWARD_DAYS, min_periods=20).max()[::-1].shift(-1) / c
    return f


STAGE_NAME = {5: "눌림 지지 확인", 4: "지지 확인 중", 3: "바닥 다지기", 2: "관망", 1: "약세", 0: "약세"}


def similar_stats(frames: dict[str, pd.DataFrame]) -> dict[int, dict]:
    """종목군 전체에서 같은 단계였던 날 이후 60거래일 최고가 배수 통계"""
    rows = []
    for f in frames.values():
        ok = f.dropna(subset=["fwd"])
        ok = ok.iloc[20:-1] if len(ok) > 25 else ok.iloc[0:0]
        rows.append(ok[["stage", "fwd"]])
    if not rows:
        return {}
    allr = pd.concat(rows)
    allr = allr[np.isfinite(allr["fwd"])]
    out = {}
    for s in range(6):
        x = allr[allr["stage"] == s]["fwd"]
        if len(x) >= 30:
            out[s] = {"n": int(len(x)), "median": float(x.median()),
                      "hit20": float((x >= 1.2).mean() * 100), "hit50": float((x >= 1.5).mean() * 100)}
    return out


# ───────────────── 5. AI 종합판단 (선택) ─────────────────
def ai_summary(item: dict) -> str:
    if not ANTHROPIC_KEY:
        return ""
    facts = {k: item[k] for k in ("t", "name", "price", "strength", "runway", "tags", "ddays", "stage")}
    facts["filings"] = [f"{f['date']} {f['form']} {f['items']}" for f in item["filings"][:15]]
    prompt = ("다음은 미국 소형주의 SEC 공시·재무 요약이다. 회사가 주가를 부양해야 할 동기(상장유지, 현금조달, "
              "물량출회, 병합)와 위험(희석, 출회)을 한국어 3문장 이내로 판단하라. 확실하지 않은 건 '확인 필요'로 표시하고 "
              "매수 권유는 하지 마라.\n" + json.dumps(facts, ensure_ascii=False))
    try:
        r = http.post("https://api.anthropic.com/v1/messages", timeout=60,
                      headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": AI_MODEL, "max_tokens": 400,
                            "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    except Exception as e:
        log.warning("AI 요약 실패 %s: %s", item["t"], e)
        return ""


# ───────────────── 6. 조립 ─────────────────
def analyze() -> list[dict]:
    today = datetime.now(KST).date()
    uni = universe()
    if not uni:
        return []
    prices = daily([u["t"] for u in uni])
    frames = {t: tech_frame(df) for t, df in prices.items()}
    stats = similar_stats(frames)
    ciks = cik_map()
    items = []
    for u in uni:
        t = u["t"]
        try:
            df = prices.get(t)
            price = float(df["Close"].iloc[-1]) if df is not None else (u["price"] or 0)
            cik = ciks.get(t)
            fs = filings(cik) if cik else []
            fin = financials(cik) if cik else {}
            tags, ddays, raw = motive_tags(price, u["float"], fs, fin, df, today)
            stg = int(frames[t]["stage"].iloc[-1]) if t in frames else 0
            st = stats.get(stg, {})
            mult = st.get("median")
            stop = float(df["Low"].iloc[-5:].min() * 0.97) if df is not None else None
            s_pts = strength(raw) * 20
            t_pts = stg * 20
            e_pts = st.get("hit20", 0)
            total = round(0.45 * s_pts + 0.35 * t_pts + 0.20 * e_pts, 1)
            items.append({
                "t": t, "name": u["name"], "price": price, "chg": u["chg"], "mcap": u["mcap"],
                "float": u["float"], "cik": cik, "strength": strength(raw), "raw": raw,
                "runway": fin.get("runway"), "cash": fin.get("cash"), "burn": fin.get("burn"),
                "equity": fin.get("equity"), "tags": tags, "ddays": ddays,
                "stage": stg, "stage_name": STAGE_NAME[stg],
                "checks": {k: bool(frames[t][k].iloc[-1]) for k in ("ma20", "lows", "upvol", "liq", "cool")}
                if t in frames else {},
                "similar": st, "target": price * mult if mult else None, "stop": stop, "total": total,
                "filings": [dict(f, ko=FORM_KO.get(f"{f['form']}:{f['items'][:4]}" if f["form"] == "8-K"
                                                   else f["form"], "")) for f in fs
                            if f["form"] in ("8-K", "424B5", "424B4", "424B3", "S-1", "S-1/A", "S-3", "S-3/A",
                                             "EFFECT", "DEF 14A", "PRE 14A", "NT 10-Q", "NT 10-K", "10-Q", "10-K")
                            ][:20],
                "ai": "",
            })
        except Exception as e:
            log.warning("%s 분석 실패: %s", t, e)
    items.sort(key=lambda x: x["total"], reverse=True)
    for it in items[:AI_TOP]:
        it["ai"] = ai_summary(it)
    return items


# ───────────────── 7. 출력 ─────────────────
def write_page(items: list[dict]) -> None:
    tpl = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    payload = {"updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"), "items": items,
               "forward": FORWARD_DAYS}
    data = json.dumps(payload, ensure_ascii=False, default=lambda o: None).replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tpl.replace("/*__DATA__*/null", data), encoding="utf-8")
    (OUT.parent / ".nojekyll").write_text("")
    log.info("대시보드 저장: %s (%d종목)", OUT, len(items))


def telegram(items: list[dict]) -> None:
    if not TOKEN:
        return
    try:
        chat = json.loads((ROOT / "state.json").read_text(encoding="utf-8")).get("chat_id")
    except Exception:
        chat = os.getenv("TELEGRAM_CHAT_ID")
    if not chat:
        return
    rows = []
    for i, it in enumerate(items[:TOP_TELEGRAM], 1):
        p = f"{it['price']:.3f}" if it["price"] < 1 else f"{it['price']:.2f}"
        rw = "-" if it["runway"] is None else ("흑자" if it["runway"] >= 99 else f"{it['runway']:.0f}개월")
        rows.append(f"{i:<2} {it['t']:<5} {p:>6} {it['strength']}/5 {it['stage']}/5 {it['total']:>4.0f}")
    body = "\n".join(["#  티커    주가 강도 차트 통합", "─" * 30] + rows)
    link = f'\n\n📊 <a href="{PAGES_URL}">전체 대시보드 보기</a>' if PAGES_URL else ""
    text = (f"🎯 <b>펌핑 목적 워치리스트</b> ({datetime.now(KST):%m/%d %H:%M})\n"
            f"<pre>{html.escape(body)}</pre>강도: 부양 동기 · 차트: 진입 조건{link}\n⚠️ 확률 아님 · 투자 판단은 본인 책임")
    try:
        http.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", timeout=15,
                  data={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"})
    except Exception as e:
        log.warning("텔레그램 실패: %s", e)


def main():
    if not SEC_EMAIL:
        log.warning("SEC_EMAIL 이 없습니다 — SEC 접속이 차단될 수 있어요 (GitHub Secrets 에 등록)")
    items = analyze()
    if not items:
        log.error("분석 결과 없음")
        return
    write_page(items)
    telegram(items)


if __name__ == "__main__":
    main()
