"""
펌핑 목적 워치리스트 + 통합점수 대시보드 생성기
════════════════════════════════════════════
1) 나스닥 저가 소형주 수집 (TradingView 스캐너) + 텔레그램 /add 관심종목
2) SEC 공시·재무 분석 → '주가를 띄워야 할 이유' 태그
     상장유지 · 현금조달 · 물량출회 · 병합 · 매집의심 · 소형유통
3) 차트 진입 조건 n/5 + 같은 종목군 과거 유사 사례로 예상 고점 배수 추산
4) 오늘 1분봉 신호 요약: 과매도 · 신저가 · 장중 반등 · 거래량 폭증 · 유동성/변동성 주의
5) docs/index.html (GitHub Pages) 생성 + 텔레그램 요약 전송

환경변수
  TELEGRAM_TOKEN  : 텔레그램 봇 토큰
  SEC_EMAIL       : SEC 접속용 연락 이메일 (SEC 규정상 필수)
  PAGES_URL       : 대시보드 주소 (텔레그램 링크용)
  ANTHROPIC_API_KEY (선택): 있으면 상위 종목에 AI 종합판단 추가 (유료)
"""

import html
import json
import logging
import math
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
MIN_PRICE, MAX_PRICE = 0.01, 5.0
MAX_MCAP = 300e6                  # 시총 상한 (하한 없음 — 초소형주도 포함)
MIN_AVG_VOL = 100_000
MAX_UNIVERSE = 400                # 평균 거래량 상위
TODAY_TOP = 100                   # + 오늘 거래량 상위 (당일 급등주 포함용)
FORWARD_DAYS = 60
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
ET = ZoneInfo("America/New_York")
# ═══════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("watchlist")
http = requests.Session()
UA_WEB = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36"}


def num(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def rnd(v, d=4):
    v = num(v)
    return None if v is None else round(v, d)


def sig(v, n=5):
    """유효숫자 n자리 (저가주 소수점 보존)"""
    v = num(v)
    if v is None or v == 0:
        return v
    return round(v, max(0, n - 1 - int(math.floor(math.log10(abs(v))))))


# ───────────────── 1. 종목 수집 ─────────────────
REQ_COLS = ["description", "close", "change", "average_volume_10d_calc", "volume"]
OPT_COLS = ["market_cap_basic", "float_shares_outstanding", "bid", "ask"]


def tv_scan(filters: list, sort: str, n: int, tickers: list[str] | None = None) -> list[dict]:
    for cols in (REQ_COLS + OPT_COLS, REQ_COLS):        # 선택 컬럼 오류 시 필수만으로 재시도
        body = {"markets": ["america"], "filter": filters, "columns": cols,
                "sort": {"sortBy": sort, "sortOrder": "desc"}, "range": [0, n]}
        if tickers:
            body["symbols"] = {"query": {"types": []}, "tickers": tickers}
        try:
            data = http.post("https://scanner.tradingview.com/america/scan", json=body,
                             headers=UA_WEB, timeout=30).json()
            if "data" not in data:
                raise ValueError(str(data)[:150])
            out = []
            for row in data["data"]:
                d = dict(zip(cols, row["d"]))
                out.append({"t": row["s"].split(":")[-1].replace(".", "-"), "name": d.get("description") or "",
                            "price": num(d.get("close")), "chg": num(d.get("change")),
                            "vol": num(d.get("volume")), "avgvol": num(d.get("average_volume_10d_calc")),
                            "mcap": num(d.get("market_cap_basic")), "float": num(d.get("float_shares_outstanding")),
                            "bid": num(d.get("bid")), "ask": num(d.get("ask"))})
            return out
        except Exception as e:
            log.warning("스캐너 실패 (%s): %s", "전체컬럼" if cols != REQ_COLS else "필수컬럼", e)
            time.sleep(2)
    return []


def watch_tickers() -> list[str]:
    try:
        return json.loads((ROOT / "state.json").read_text(encoding="utf-8")).get("tickers", [])
    except Exception:
        return []


def universe() -> list[dict]:
    base = [{"left": "exchange", "operation": "in_range", "right": EXCHANGES},
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "in_range", "right": [MIN_PRICE, MAX_PRICE]},
            {"left": "average_volume_10d_calc", "operation": "greater", "right": MIN_AVG_VOL}]
    rows = tv_scan(base, "average_volume_10d_calc", MAX_UNIVERSE) + tv_scan(base, "volume", TODAY_TOP)
    rows = [r for r in rows if r["mcap"] is None or r["mcap"] <= MAX_MCAP]
    watch = watch_tickers()
    if watch:   # 텔레그램 관심종목은 조건과 무관하게 포함
        sym = [f"{ex}:{t.replace('-', '.')}" for t in watch for ex in ("NASDAQ", "NYSE", "AMEX")]
        for r in tv_scan([], "volume", len(sym), tickers=sym):
            r["watch"] = True
            rows.insert(0, r)
    seen, out = set(), []
    for r in rows:
        if r["t"] not in seen:
            seen.add(r["t"])
            out.append(r)
    log.info("대상 종목 %d개 (관심종목 %d)", len(out), sum(1 for r in out if r.get("watch")))
    return out


def _download(tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            df = yf.download(chunk, group_by="ticker", auto_adjust=False, threads=True, progress=False, **kw)
        except Exception as e:
            log.warning("시세 다운로드 실패: %s", e)
            continue
        if df is None or df.empty:
            continue
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(subset=["Close"])
                if len(sub):
                    out[t] = sub
            except KeyError:
                pass
        time.sleep(1)
    return out


def daily(tickers):
    return {t: d for t, d in _download(tickers, period="1y", interval="1d", actions=True).items() if len(d) >= 30}


def intraday(tickers):
    """가장 최근 거래일 1분봉 (프리·애프터 포함)"""
    out = {}
    for t, m in _download(tickers, period="1d", interval="1m", prepost=True).items():
        try:
            idx = m.index.tz_convert(ET) if m.index.tz is not None else m.index.tz_localize("UTC").tz_convert(ET)
            m = m.set_axis(idx)
            m = m[m.index.date == m.index[-1].date()]
            if len(m) >= 5:
                out[t] = m
        except Exception:
            continue
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


def motive_tags(price, fl, fs, fin, df, today):
    tags, ddays, raw = [], [], 0

    def tag(name, level, note=""):
        tags.append({"name": name, "level": level, "note": note})

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

    resale = has(fs, today, ["424B3"], 180) or has(fs, today, ["8-K"], 180, "3.02")
    if resale:
        tag("물량 출회", "예정", f"{resale['form']} {resale['date']}"); raw += 1

    proxy = has(fs, today, ["PRE 14A", "DEF 14A"], 180)
    if proxy and price < 1:
        tag("병합", "안건 가능", f"{proxy['form']} {proxy['date']}"); raw += 1
    rs = 0
    if df is not None and "Stock Splits" in df:
        sp = df["Stock Splits"]
        rs = int(((sp > 0) & (sp < 1)).sum())
    if rs:
        tag("병합 이력", f"{rs}회", "최근 1년")

    if has(fs, today, ["NT 10-Q", "NT 10-K"], 120):
        tag("공시 지연", "위험")

    if fl and fl < 10e6:
        tag("소형 유통", f"{fl / 1e6:.1f}M"); raw += 1

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
def rsi_series(c: pd.Series, n: int = 14) -> pd.Series:
    delta = c.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ls = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = g / ls.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(ls != 0, 100.0)


def tech_frame(df: pd.DataFrame) -> pd.DataFrame:
    c, l, v = df["Close"], df["Low"], df["Volume"]
    ma20 = c.rolling(20).mean()
    gap = (c / ma20 - 1) * 100
    up = c.diff() > 0
    upv = (v * up).rolling(10).sum()
    dnv = (v * (~up)).rolling(10).sum()
    rsi = rsi_series(c)
    f = pd.DataFrame(index=df.index)
    f["ma20"] = (gap >= 0) & (gap <= 15)
    f["lows"] = l.rolling(3).min() > l.shift(3).rolling(3).min()
    f["upvol"] = upv > dnv * 1.2
    f["liq"] = (c * v).rolling(20).mean() >= 300_000
    f["cool"] = ((c / c.shift(5) - 1) < 0.5) & (rsi.fillna(50) < 75)
    f["stage"] = f[["ma20", "lows", "upvol", "liq", "cool"]].sum(axis=1)
    f["fwd"] = df["High"][::-1].rolling(FORWARD_DAYS, min_periods=20).max()[::-1].shift(-1) / c
    f["rsi"] = rsi
    return f


STAGE_NAME = {5: "눌림 지지 확인", 4: "지지 확인 중", 3: "바닥 다지기", 2: "관망", 1: "약세", 0: "약세"}


def similar_stats(frames):
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


# ───────────────── 5. 오늘 신호 요약 (1분봉) ─────────────────
def fmt_p(v):
    return "-" if v is None else (f"{v:.4f}" if v < 1 else f"{v:.2f}")


def fmt_money(v):
    if v is None:
        return "-"
    return f"{v / 1e6:.1f}M 달러" if v >= 1e6 else f"{v / 1e4:.1f}만 달러"


def snapshot(u: dict, d: pd.DataFrame | None, m: pd.DataFrame | None) -> dict:
    """보내주신 화면처럼: 현재가·RSI·회전율·스프레드·당일/52주 위치·신호 배지"""
    s = {"signals": [], "keys": []}
    if m is not None and len(m):
        price = float(m["Close"].iloc[-1])
        day = m.index[-1].date()
        reg = m[m.index.time >= datetime.strptime("09:30", "%H:%M").time()]
        s["open"] = float((reg if len(reg) else m)["Open"].iloc[0])
        s["day_hi"], s["day_lo"] = float(m["High"].max()), float(m["Low"].min())
        vol_day = float(m["Volume"].sum())
        tp = (m["High"] + m["Low"] + m["Close"]) / 3
        s["vwap"] = float((tp * m["Volume"]).sum() / max(vol_day, 1)) if vol_day else None
        r = rsi_series(m["Close"]).iloc[-1] if len(m) >= 20 else np.nan
        s["rsi_src"] = "1분봉"
    else:
        price = u["price"] or (float(d["Close"].iloc[-1]) if d is not None else 0)
        day = datetime.now(ET).date()
        s["open"] = s["day_hi"] = s["day_lo"] = s["vwap"] = None
        vol_day = u["vol"] or 0
        r = np.nan
    if (r is None or np.isnan(r)) and d is not None:
        r = rsi_series(d["Close"]).iloc[-1]
        s["rsi_src"] = "일봉"
    s["rsi"] = rnd(r, 2)
    s["price"] = price

    prev = None
    if d is not None and len(d):
        before = d[[ts.date() < day for ts in d.index]]
        if len(before):
            prev = float(before["Close"].iloc[-1])
        lo52, hi52 = float(d["Low"].min()), float(d["High"].max())
        if s["day_lo"]:
            lo52, hi52 = min(lo52, s["day_lo"]), max(hi52, s["day_hi"])
        s["lo52"], s["hi52"] = lo52, hi52
    s["prev"] = prev
    s["chg"] = (price / prev - 1) * 100 if prev else u.get("chg")

    bid, ask = u.get("bid"), u.get("ask")
    s["bid"], s["ask"] = bid, ask
    s["spread"] = (ask - bid) / ((ask + bid) / 2) * 100 if bid and ask and ask >= bid else None
    shares = u.get("float") or ((u["mcap"] / price) if u.get("mcap") and price else None)
    s["shares"] = shares
    s["vol_day"] = vol_day
    s["turnover"] = vol_day / shares if shares and vol_day else None
    s["relvol"] = vol_day / u["avgvol"] if u.get("avgvol") and vol_day else None
    s["mcap"] = u.get("mcap")

    # 1분봉 스파크라인 (최대 120점)
    if m is not None and len(m) >= 5:
        step = max(1, len(m) // 120)
        pts = m["Close"].iloc[::step]
        if pts.index[-1] != m.index[-1]:
            pts = pd.concat([pts, m["Close"].iloc[-1:]])
        k = lambda ts: ts.astimezone(KST).strftime("%H:%M")
        s["spark"] = {"p": [sig(v) for v in pts], "t0": k(m.index[0]), "t1": k(m.index[-1]),
                      "tm": k(m.index[len(m) // 2]), "date": f"{day.month}/{day.day}"}

    # ── 신호 배지 ──
    def add(key, level, text):
        s["signals"].append({"k": key, "lv": level, "t": text})
        s["keys"].append(key)

    rv = s["rsi"]
    if rv is not None and rv <= 30:
        add("과매도", "red", f"RSI {rv:.1f}({s['rsi_src']}), 단기 낙폭 과대 구간")
    elif rv is not None and rv >= 75:
        add("과열", "org", f"RSI {rv:.1f}({s['rsi_src']}), 단기 급등 과열 구간")
    if s.get("lo52") and s["day_lo"] and s["day_lo"] <= s["lo52"] * 1.005:
        add("신저가", "red", "당일 저가가 52주 저가와 일치")
    elif s.get("hi52") and price >= s["hi52"] * 0.97:
        add("신고가 근접", "grn", f"52주 고가 {fmt_p(s['hi52'])}의 97% 이상")
    if s["day_lo"] and s["open"] and price >= s["day_lo"] * 1.05 and price > s["open"]:
        add("장중 반등", "grn", f"저가 대비 +{(price / s['day_lo'] - 1) * 100:.2f}%, "
                                f"시가 대비 +{(price / s['open'] - 1) * 100:.2f}%")
    elif s["day_hi"] and price <= s["day_hi"] * 0.85:
        add("고점 대비 밀림", "org", f"당일 고가 대비 -{(1 - price / s['day_hi']) * 100:.1f}%")
    if s["relvol"] and s["relvol"] >= 5:
        add("거래량 폭증", "vio", f"평소(10일 평균)의 {s['relvol']:.1f}배")
    if s["vwap"]:
        if price > s["vwap"] * 1.02:
            add("VWAP 위", "grn", f"VWAP {fmt_p(s['vwap'])} 대비 +{(price / s['vwap'] - 1) * 100:.1f}%")
        elif price < s["vwap"] * 0.98:
            add("VWAP 아래", "org", f"VWAP {fmt_p(s['vwap'])} 대비 -{(1 - price / s['vwap']) * 100:.1f}%")
    dollar = price * vol_day if vol_day else 0
    if (s["spread"] and s["spread"] >= 2) or dollar < 200_000:
        bits = []
        if s["spread"]:
            bits.append(f"스프레드 {s['spread']:.1f}%대")
        if dollar < 200_000:
            bits.append(f"거래대금 ${dollar / 1e3:.0f}K")
        add("유동성 주의", "org", ", ".join(bits) or "호가 얇음")
    if (s["turnover"] and s["turnover"] >= 3) or (s["mcap"] and s["mcap"] < 5e6):
        bits = []
        if s["mcap"]:
            bits.append(f"시총 약 {fmt_money(s['mcap'])}")
        if s["turnover"]:
            bits.append(f"회전율 {s['turnover']:.1f}배")
        add("변동성 주의", "org", ", ".join(bits))

    for k in ("price", "open", "day_hi", "day_lo", "vwap", "prev", "lo52", "hi52", "bid", "ask"):
        if k in s:
            s[k] = sig(s[k], 6)
    for k in ("chg", "spread", "turnover", "relvol"):
        s[k] = rnd(s.get(k), 2)
    return s


# ───────────────── 6. AI 종합판단 (선택) ─────────────────
def ai_summary(item: dict) -> str:
    if not ANTHROPIC_KEY:
        return ""
    facts = {k: item[k] for k in ("t", "name", "price", "strength", "runway", "tags", "ddays", "stage")}
    facts["filings"] = [f"{f['date']} {f['form']} {f['items']}" for f in item["filings"][:15]]
    facts["today"] = [x["k"] + ": " + x["t"] for x in item["snap"]["signals"]]
    prompt = ("다음은 미국 소형주의 SEC 공시·재무·오늘 시세 요약이다. 회사가 주가를 부양해야 할 동기(상장유지, 현금조달, "
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


# ───────────────── 7. 조립 ─────────────────
FILING_FORMS = ("8-K", "424B5", "424B4", "424B3", "S-1", "S-1/A", "S-3", "S-3/A",
                "EFFECT", "DEF 14A", "PRE 14A", "NT 10-Q", "NT 10-K", "10-Q", "10-K")


def analyze() -> list[dict]:
    today = datetime.now(KST).date()
    uni = universe()
    if not uni:
        return []
    tickers = [u["t"] for u in uni]
    prices = daily(tickers)
    mins = intraday(tickers)
    log.info("일봉 %d · 1분봉 %d", len(prices), len(mins))
    frames = {t: tech_frame(df) for t, df in prices.items()}
    stats = similar_stats(frames)
    ciks = cik_map()
    items = []
    for u in uni:
        t = u["t"]
        try:
            df = prices.get(t)
            snap = snapshot(u, df, mins.get(t))
            price = snap["price"] or u["price"] or 0
            cik = ciks.get(t)
            fs = filings(cik) if cik else []
            fin = financials(cik) if cik else {}
            tags, ddays, raw = motive_tags(price, u["float"], fs, fin, df, today)
            stg = int(frames[t]["stage"].iloc[-1]) if t in frames else 0
            st = stats.get(stg, {})
            mult = st.get("median")
            stop = float(df["Low"].iloc[-5:].min() * 0.97) if df is not None else None
            total = round(0.45 * strength(raw) * 20 + 0.35 * stg * 20 + 0.20 * st.get("hit20", 0), 1)
            items.append({
                "t": t, "name": u["name"], "price": price, "chg": snap["chg"], "mcap": u["mcap"],
                "float": u["float"], "cik": cik, "watch": bool(u.get("watch")),
                "strength": strength(raw), "raw": raw,
                "runway": fin.get("runway"), "cash": fin.get("cash"), "burn": fin.get("burn"),
                "equity": fin.get("equity"), "tags": tags, "ddays": ddays,
                "stage": stg, "stage_name": STAGE_NAME[stg],
                "checks": {k: bool(frames[t][k].iloc[-1]) for k in ("ma20", "lows", "upvol", "liq", "cool")}
                if t in frames else {},
                "similar": st, "target": price * mult if mult and mult > 1.02 else None, "stop": stop, "total": total,
                "snap": snap,
                "filings": [dict(f, ko=FORM_KO.get(f"{f['form']}:{f['items'][:4]}" if f["form"] == "8-K"
                                                   else f["form"], "")) for f in fs if f["form"] in FILING_FORMS][:20],
                "ai": "",
            })
        except Exception as e:
            log.warning("%s 분석 실패: %s", t, e)
    items.sort(key=lambda x: x["total"], reverse=True)
    for it in items[:AI_TOP]:
        it["ai"] = ai_summary(it)
    return items


# ───────────────── 8. 출력 ─────────────────
def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_page(items: list[dict]) -> None:
    tpl = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    payload = _clean({"updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"), "items": items,
                      "forward": FORWARD_DAYS})
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
