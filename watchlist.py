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
import pickle
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import report

# ═════════════════════ 설정 ═════════════════════
EXCHANGES = ["NASDAQ"]
MIN_PRICE, MAX_PRICE = 0.01, 5.0
MAX_MCAP = 300e6                  # 시총 상한 (하한 없음 — 초소형주도 포함)
MIN_AVG_VOL = 100_000
MAX_UNIVERSE = 400                # 평균 거래량 상위
TODAY_TOP = 100                   # + 오늘 거래량 상위 (당일 급등주 포함용)
FORWARD_DAYS = 60
SURGE_REG = 0.20                  # 본장 급등 기준: 전일 종가 대비 고가 +20% 터치
SURGE_PRE = 0.10                  # 프리장 급등 기준: 전일 종가 대비 프리장 고가 +10% 터치
PRIOR_K = 20                      # 표본 적은 조건은 상위 조건 확률 쪽으로 보정
HIST_PERIOD = "5y"                # 본장 확률 학습 기간 (일봉)
PRE_PERIOD = "700d"               # 프리장 확률 학습 기간 (1시간봉 무료 한도 730일보다 살짝 짧게)
TOP_TELEGRAM = 10
SURGE_TOP = 8                     # 금일 급등 후보 개수
JB_TOP = 8                        # 종배 예상 개수
JB_HIT = 0.10                     # 종배 성공 기준: 다음날 고가 +10% 터치
AI_TOP = 10
AI_MODEL = "claude-sonnet-5-5"

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
SEC_EMAIL = os.getenv("SEC_EMAIL", "").strip()
PAGES_URL = os.getenv("PAGES_URL", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()          # 무료 (하루 1,000회)
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()
GROQ_PREFER = ["gpt-oss-120b", "llama-4-maverick", "llama-3.3-70b", "70b", "gpt-oss-20b", "llama"]
AI_PER_RUN = 40                                            # 한 번 실행에 새로 쓰는 AI 판단 수 (무료 한도 보호)
ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "index.html"
CACHE = ROOT / ".cache"            # 하루 한 번 받는 큰 데이터 (GitHub actions/cache 로 보관)
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
    return {t: d for t, d in _download(tickers, period=HIST_PERIOD, interval="1d", actions=True).items() if len(d) >= 30}


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


def shares_hist(cik: int) -> dict | None:
    """발행주식수 (최신 vs 약 5개월+ 전) — 실제로 얼마나 희석됐는지"""
    d = sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json")
    pts = [f for f in (d or {}).get("units", {}).get("shares", []) if "end" in f]
    if not pts:
        return None
    pts.sort(key=lambda f: (f["end"], f.get("filed", "")))
    now = pts[-1]
    end = date.fromisoformat(now["end"])
    older = [f for f in pts if date.fromisoformat(f["end"]) <= end - timedelta(days=150)]
    prev = older[-1] if older else None
    g = (now["val"] / prev["val"] - 1) * 100 if prev and prev["val"] else None
    return {"now": now["val"], "now_date": now["end"], "prev": prev["val"] if prev else None,
            "prev_date": prev["end"] if prev else None, "growth": None if g is None else round(g, 1)}


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
    try:
        sh = shares_hist(cik)
    except Exception:
        sh = None
    return {"sh": sh, "cash": cash[0] if cash else None, "cash_date": cash[1] if cash else None,
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
        sp = df["Stock Splits"].iloc[-252:]
        rs = int(((sp > 0) & (sp < 1)).sum())
    if rs:
        tag("병합 이력", f"{rs}회", "최근 1년")

    if has(fs, today, ["NT 10-Q", "NT 10-K"], 120):
        tag("공시 지연", "위험")

    if fl and fl < 10e6:
        tag("소형 유통", f"{fl / 1e6:.1f}M"); raw += 1

    if df is not None and len(df) > 30:
        df = df.iloc[-60:]
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
    # 급등 확률용: 그날 거래량 배수·등락률·종가 → 다음날 본장 고가가 +20% 터치했나
    f["relvol"] = v / v.rolling(20).mean().shift(1)
    f["chg"] = c.pct_change() * 100
    f["close"] = c
    f["yreg"] = (df["High"].shift(-1) / c - 1 >= SURGE_REG).astype(float).where(df["High"].shift(-1).notna())
    f["greg"] = df["High"].shift(-1) / c - 1            # 다음날 고가 상승률
    f["gap"] = df["Open"].shift(-1) / c - 1             # 다음날 시가 갭
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    f["cp"] = ((c - df["Low"]) / rng).fillna(0.5)        # 종가가 그날 범위 어디서 끝났나 (1=고가 마감)
    f["high"], f["low"] = df["High"], df["Low"]
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
        reg = m[m.index.time >= dtime(9, 30)]
        s["open"] = float((reg if len(reg) else m)["Open"].iloc[0])
        s["day_hi"], s["day_lo"] = float(m["High"].max()), float(m["Low"].min())
        vol_day = float(m["Volume"].sum())
        tp = (m["High"] + m["Low"] + m["Close"]) / 3
        s["vwap"] = float((tp * m["Volume"]).sum() / max(vol_day, 1)) if vol_day else None
        t_ = m.index.time
        pre_m, reg_m = m[t_ < dtime(9, 30)], m[(t_ >= dtime(9, 30)) & (t_ < dtime(16, 0))]
        s["pre_hi"] = float(pre_m["High"].max()) if len(pre_m) else None
        s["reg_hi"] = float(reg_m["High"].max()) if len(reg_m) else None
        s["reg_done"] = bool(len(reg_m)) and reg_m.index[-1].time() >= dtime(15, 55)
        if len(reg_m):
            s["reg_last"], s["reg_lo"] = float(reg_m["Close"].iloc[-1]), float(reg_m["Low"].min())
            s["reg_vol"] = float(reg_m["Volume"].sum())
            s["reg_time"] = reg_m.index[-1].astimezone(KST).strftime("%H:%M")
        s["sdate"] = day.isoformat()
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
        lo52, hi52 = float(d["Low"].iloc[-252:].min()), float(d["High"].iloc[-252:].max())
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

    for k in ("price", "open", "day_hi", "day_lo", "vwap", "prev", "lo52", "hi52", "bid", "ask", "pre_hi", "reg_hi",
              "reg_last", "reg_lo"):
        if k in s:
            s[k] = sig(s[k], 6)
    for k in ("chg", "spread", "turnover", "relvol"):
        s[k] = rnd(s.get(k), 2)
    return s


# ───────────────── 5-2. 금일 프리장·본장 급등 확률 (과거 실제 빈도) ─────────────────
RV_EDGES = [1.5, 3, 10]                 # 전날 거래량 배수 구간
CH_EDGES = [-10, -3, 3, 10, 30]         # 전날 등락률 구간
PX_EDGES = [0.5, 1]                     # 주가 구간
WEEK = "월화수목금토일"


def fkeys(relvol, chg, close) -> list[tuple]:
    rv = int(np.searchsorted(RV_EDGES, relvol, side="right")) if relvol == relvol else -1
    ch = int(np.searchsorted(CH_EDGES, chg, side="right")) if chg == chg else -1
    px = int(np.searchsorted(PX_EDGES, close, side="right"))
    return [(), (rv,), (rv, ch), (rv, ch, px)]          # 넓은 조건 → 좁은 조건


def jbkeys(relvol, chg, close, cp) -> list[tuple]:
    """종배용: 오늘 거래량·등락·종가 위치(고가 마감인지)·주가대"""
    k = fkeys(relvol, chg, close)
    cb = 0 if cp < 0.3 else 1 if cp < 0.7 else 2
    return [(), ("j",) + k[1], ("j",) + k[2], ("j",) + k[2] + (cb,), ("j",) + k[3] + (cb,)]


class Rate:
    """조건별 '다음날 급등' 실제 빈도. 표본 적은 조건은 상위 조건 쪽으로 보정
    c[key] = [표본수, 적중수, 고가상승합, 적중시 고가상승합, 갭상승수, 갭합]"""
    def __init__(self):
        self.c = defaultdict(lambda: [0, 0, 0.0, 0.0, 0, 0.0])

    def add(self, ks, y, g=0.0, up=0, gap=0.0):
        g, gap = max(-0.9, min(g, 3.0)), max(-0.9, min(gap, 3.0))
        for k in ks:
            c = self.c[k]
            c[0] += 1; c[1] += y; c[2] += g; c[3] += g if y else 0.0; c[4] += up; c[5] += gap

    def stats(self, ks, min_n=30) -> dict:
        """가장 좁은 조건(표본 30 이상)의 평균 고가 상승·터졌을 때 평균·갭상승 비율"""
        for k in reversed(ks):
            if k in self.c and self.c[k][0] >= min_n:
                n, h, sg, sgh, up, gap = self.c[k]
                return {"avg_hi": round(sg / n * 100, 1), "hit_hi": round(sgh / h * 100, 1) if h else None,
                        "up": round(up / n * 100, 1), "gap": round(gap / n * 100, 2), "n": n}
        return {}

    @property
    def base(self):
        n, h = self.c[()][:2]
        return h / n if n else None

    def p(self, ks):
        p, used = self.base, self.c[()][0]
        if p is None:
            return None, 0
        for k in ks[1:]:
            if k in self.c:
                n, h = self.c[k][:2]
                p = (h + PRIOR_K * p) / (n + PRIOR_K)
                used = n
        return p, used


def target_session() -> date:
    """확률을 낼 '오늘 장' 날짜 (뉴욕 기준). 장 마감 후·주말이면 다음 거래일"""
    now = datetime.now(ET)
    d = now.date()
    if now.weekday() >= 5 or now.time() >= dtime(16, 0):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d


def premarket_highs(tickers) -> dict[str, pd.Series]:
    """최근 2년 1시간봉 → 날짜별 프리장(4:00~9:30) 고가"""
    out = {}
    for t, h in _download(tickers, period=PRE_PERIOD, interval="1h", prepost=True).items():
        try:
            idx = h.index.tz_convert(ET) if h.index.tz is not None else h.index.tz_localize("UTC").tz_convert(ET)
            h = h.set_axis(idx)
            pre = h[h.index.time < dtime(9, 30)]
            if len(pre):
                out[t] = pre["High"].groupby(pre.index.date).max()
        except Exception:
            continue
    return out


def build_models(frames, pre_hi, target):
    reg, pre, jb = Rate(), Rate(), Rate()
    for t, f in frames.items():
        tr = f[f.index.date < target].iloc[:-1]          # 결과가 확정된 날만 학습
        tr = tr.dropna(subset=["relvol", "chg", "yreg", "greg", "gap"])
        for r in tr[["relvol", "chg", "close", "yreg", "greg", "gap", "cp"]].itertuples(index=False):
            reg.add(fkeys(r.relvol, r.chg, r.close), int(r.yreg), r.greg)
            jb.add(jbkeys(r.relvol, r.chg, r.close, r.cp), int(r.greg >= JB_HIT), r.greg, int(r.gap > 0), r.gap)
        ph = pre_hi.get(t)
        if ph is None or not len(f):
            continue
        d64 = f.index.values.astype("datetime64[D]")
        for D, hi in ph.items():
            if D >= target:
                continue
            i = int(np.searchsorted(d64, np.datetime64(D))) - 1   # D 전날 행
            if i < 1:
                continue
            row = f.iloc[i]
            if row["relvol"] != row["relvol"] or not row["close"]:
                continue
            pre.add(fkeys(row["relvol"], row["chg"], row["close"]), int(hi / row["close"] - 1 >= SURGE_PRE),
                    hi / row["close"] - 1)
    log.info("확률 학습: 본장 %d일 · 프리장 %d일 · 종배 %d일", reg.c[()][0], pre.c[()][0], jb.c[()][0])
    return reg, pre, jb


def predict(f, reg, pre, target, snap) -> dict | None:
    if f is None:
        return None
    rows = f[f.index.date < target]
    if not len(rows):
        return None
    r = rows.iloc[-1]
    ks = fkeys(r["relvol"], r["chg"], r["close"])
    pr, nr = reg.p(ks)
    pp, np_ = pre.p(ks)
    out = {"date": f"{target.month}/{target.day}({WEEK[target.weekday()]})",
           "reg": None if pr is None else round(pr * 100, 1), "n_reg": nr,
           "reg_base": None if reg.base is None else round(reg.base * 100, 1),
           "pre": None if pp is None else round(pp * 100, 1), "n_pre": np_,
           "pre_base": None if pre.base is None else round(pre.base * 100, 1),
           "reg_th": int(SURGE_REG * 100), "pre_th": int(SURGE_PRE * 100),
           "basis": f"전일 거래량 {r['relvol']:.1f}배 · 등락 {r['chg']:+.1f}%" if r["relvol"] == r["relvol"] else "",
           "reg_st": reg.stats(ks), "pre_st": pre.stats(ks),
           "ref": float(r["close"]), "ref_hi": float(r["high"]), "relvol": num(r["relvol"]), "pchg": num(r["chg"]),
           "cp": num(r["cp"])}
    # 이미 지난 장은 실제 결과 표시
    prev = float(r["close"])
    if snap.get("sdate") == target.isoformat() and prev:
        if snap.get("pre_hi") and datetime.now(ET).time() >= dtime(9, 30):
            out["pre_act"] = round((snap["pre_hi"] / prev - 1) * 100, 1)
        if snap.get("reg_hi") and snap.get("reg_done"):
            out["reg_act"] = round((snap["reg_hi"] / prev - 1) * 100, 1)
    return out


# ───────────────── 5-3. 금일 급등 후보 · 종배 예상 (원인 분석 + 시나리오) ─────────────────
POS_WORDS = ["beat", "beats", "tops", "surpass", "raises guidance", "raised guidance", "upgrade", "record",
             "approval", "approved", "approves", "clearance", "partnership", "collaboration", "contract",
             "awarded", "buyback", "repurchase", "breakthrough", "positive", "acquire", "acquisition of",
             "to be acquired", "merger", "strategic", "launches", "expands", "patent", "grant", "milestone",
             "agreement", "order", "wins", "ai ", "bitcoin", "crypto", "treasury"]
NEG_WORDS = ["miss", "misses", "cuts guidance", "lowers guidance", "downgrade", "lawsuit", "probe",
             "investigation", "subpoena", "recall", "offering", "dilution", "priced", "registered direct",
             "private placement", "warrant", "bankruptcy", "chapter 11", "delist", "going concern",
             "reverse split", "reverse stock split", "default", "layoff", "halt", "fraud", "resigns", "fails"]
DILUTE = ["offering", "registered direct", "private placement", "warrant", "at-the-market", "priced", "shelf"]


def _hit(text, words):
    return sum(1 for w in words if w in text)


def fetch_news(t: str, hours: int = 48) -> list[dict]:
    try:
        raw = yf.Ticker(t).news or []
    except Exception:
        return []
    now, out = datetime.now(timezone.utc), []
    for it in raw:
        try:
            c = it.get("content") or it
            title = c.get("title") or ""
            if c.get("pubDate"):
                ts = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            elif it.get("providerPublishTime"):
                ts = datetime.fromtimestamp(it["providerPublishTime"], timezone.utc)
            else:
                continue
            if not title or now - ts > timedelta(hours=hours):
                continue
            link = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                    or c.get("link") or "")
            text = (title + " " + (c.get("summary") or "")).lower()
            p, n = _hit(text, POS_WORDS), _hit(text, NEG_WORDS)
            out.append({"title": title, "link": link, "ts": ts, "when": ts.astimezone(KST).strftime("%m/%d %H:%M"),
                        "pub": (c.get("provider") or {}).get("displayName") or c.get("publisher") or "",
                        "s": "호재" if p > n else "악재" if n > p else "중립", "dilute": _hit(text, DILUTE) > 0})
        except Exception:
            continue
    out.sort(key=lambda x: x["ts"], reverse=True)
    return out


HANGUL = re.compile(r"[가-힣]")


def translate_ko(text: str) -> str:
    """구글 번역 → 실패하면 Groq/Claude 로 번역 → 그래도 실패하면 원문"""
    if not text or HANGUL.search(text):
        return text
    for attempt in range(2):
        try:
            r = http.post("https://translate.googleapis.com/translate_a/single", timeout=10,
                          params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t"},
                          data={"q": text}, headers=UA_WEB)
            out = "".join(x[0] for x in r.json()[0] if x and x[0])
            if HANGUL.search(out):
                return out
        except Exception as e:
            log.debug("구글 번역 실패: %s", e)
        time.sleep(1)
    if len(text) > 80 and (ANTHROPIC_KEY or GROQ_KEY):
        out = llm("다음 영어 회사 소개를 자연스러운 한국어 2~3문장으로 번역해. 번역문만 출력해.\n\n" + text, 400)
        if HANGUL.search(out or ""):
            return out
    return text


def short_info(t: str) -> dict:
    try:
        i = yf.Ticker(t).info or {}
    except Exception:
        return {}
    si = num(i.get("shortPercentOfFloat"))
    d = i.get("dateShortInterest")
    return {"si": None if si is None else si * 100, "dtc": num(i.get("shortRatio")),
            "short": num(i.get("sharesShort")), "short_prev": num(i.get("sharesShortPriorMonth")),
            "si_date": datetime.fromtimestamp(d, timezone.utc).strftime("%m/%d") if d else None}


def enrich(t: str) -> dict:
    news = fetch_news(t)
    for n in news[:3]:
        n["ko"] = translate_ko(n["title"])
    return {"news": [{k: n.get(k) for k in ("ko", "title", "link", "when", "pub", "s", "dilute")} for n in news[:3]],
            "pos": sum(n["s"] == "호재" for n in news), "neg": sum(n["s"] == "악재" for n in news),
            "dilute_news": any(n["dilute"] for n in news), **short_info(t)}


def resistances(df: pd.DataFrame | None, ref: float) -> list[float]:
    if df is None or not ref:
        return []
    hs = sorted(set(round(float(x), 6) for x in df["High"].iloc[-126:] if x > ref * 1.08))
    out = []
    for h in hs:
        if not out or h > out[-1] * 1.10:
            out.append(h)
        if len(out) == 2:
            break
    return out


def P(v):
    return "-" if v is None else (f"${v:.4f}" if v < 1 else f"${v:.2f}")


def pct(a, b):
    return f"{(a / b - 1) * 100:+.0f}%" if a and b else ""


RISK_TAGS = {"공모 준비": 8, "ATM·공모": 8, "물량 출회": 8, "병합": 4, "공시 지연": 5}


def drivers_for(it: dict, ex: dict, today_side: bool) -> tuple[list, list]:
    """→ ([(유형, 점수, 근거)], [(리스크, 감점)])"""
    sn, p, tags = it.get("snap") or {}, it.get("prob") or {}, {t["name"]: t for t in it["tags"]}
    d, r = [], []
    j = it.get("jb") or {}
    relvol = p.get("relvol") if not today_side else j.get("relvol")
    chg = p.get("pchg") if not today_side else j.get("chg")
    if ex.get("pos"):
        n0 = next((n for n in ex.get("news", []) if n["s"] == "호재"), None)
        d.append(("호재 뉴스형", 16, f"호재 {ex['pos']}건" + (f": {n0['ko'][:40]}" if n0 and n0.get("ko") else "")))
    si = ex.get("si")
    if si and si >= 10:
        pts = 16 if si >= 20 else 9
        ev = f"공매도 유통 대비 {si:.0f}%"
        if ex.get("dtc"):
            ev += f" · 커버 {ex['dtc']:.1f}일"
            pts += 3 if ex["dtc"] >= 3 else 0
        if ex.get("short") and ex.get("short_prev") and ex["short"] > ex["short_prev"] * 1.1:
            ev += " · 공매도 증가"
            pts += 2
        if chg and chg > 5:
            pts += 3
            ev += " · 주가 상승 중(숏커버 압력)"
        d.append(("숏스퀴즈형", pts, ev + (f" ({ex['si_date']} 기준)" if ex.get("si_date") else "")))
    fl = it.get("float")
    if fl and fl < 10e6 and relvol and relvol >= 3:
        d.append(("저유통 모멘텀형", 11, f"유통 {fl / 1e6:.1f}M주 · 거래량 {relvol:.1f}배"))
    cpv = j.get("cp") if today_side else p.get("cp")
    if not today_side and chg is not None and chg >= 20 and (cpv if cpv is not None else 0.8) >= 0.7 and (relvol or 0) >= 2:
        d.append(("전일 강세 연속형", 10, f"{chg:+.0f}% · 고가권 마감 · 거래량 {relvol:.1f}배"))
    if "상장 유지" in tags and tags["상장 유지"]["level"] in ("기한형", "필요"):
        dd = next((x for x in it.get("ddays", []) if "회복" in x["label"]), None)
        d.append(("상장유지 부양형", 7, "$1 미달 통지" + (f" · 기한 D-{dd['d']}" if dd else "")))
    if "매집 의심" in tags:
        d.append(("매집형", 7, f"거래량 {tags['매집 의심']['level']} 폭증 후 가격 눌림"))
    keys = sn.get("keys") or []
    if ("과매도" in keys or "신저가" in keys) or (chg is not None and chg <= -15):
        d.append(("과매도 반등형", 6, f"{'과매도·' if '과매도' in keys else ''}{'신저가·' if '신저가' in keys else ''}등락 {chg:+.0f}%"
                  if chg is not None else "과매도"))
    for name, pts in RISK_TAGS.items():
        if name in tags:
            r.append((f"{name}({tags[name]['level']})", pts))
    if ex.get("dilute_news"):
        r.append(("증자·워런트 뉴스", 12))
    if "현금 조달" in tags and tags["현금 조달"]["level"] == "급함":
        r.append((f"현금 고갈({tags['현금 조달']['note']})", 5))
    if "고점 대비 밀림" in keys:
        r.append(("고점 대비 밀림", 5))
    if ex.get("neg"):
        r.append((f"악재 뉴스 {ex['neg']}건", 6))
    d.sort(key=lambda x: -x[1])
    return d, r


def surge_scenario(it, d, r, df) -> list[dict]:
    p, ex = it["prob"], it.get("ex", {})
    ref, ref_hi = p["ref"], p["ref_hi"]
    typ = d[0][0] if d else "확률형"
    st = p.get("reg_st") or {}
    hit = st.get("hit_hi")
    res = resistances(df, ref)
    t1 = res[0] if res else ref * (1 + SURGE_REG)
    t2 = res[1] if len(res) > 1 else (ref * (1 + hit / 100) if hit else t1 * 1.2)
    why = {"숏스퀴즈형": "숏커버링이 붙으며 매물 공백 구간 가속", "호재 뉴스형": "재료 확산으로 신규 매수 유입",
           "저유통 모멘텀형": "적은 유통 물량에 매수 몰리며 슈팅", "전일 강세 연속형": "어제 강세 흐름 이어 추가 상승",
           "상장유지 부양형": "$1 회복을 노린 부양성 매수", "매집형": "매집 물량이 슈팅 구간 진입",
           "과매도 반등형": "낙폭 과대 구간에서 기술적 반등"}.get(typ, "거래량 유입 시 단기 급등")
    lines = [{"k": "근거", "t": " · ".join(x[2] for x in d[:3]) or p.get("basis", "")},
             {"k": "강세", "t": f"프리장~장초반 전일 고가 {P(ref_hi)} 돌파 + 거래량 유지 → {why}. "
                              f"1차 {P(t1)}({pct(t1, ref)}) → 2차 {P(t2)}({pct(t2, ref)})"},
             {"k": "기본", "t": f"{P(ref)}~{P(ref_hi)} 박스 등락. VWAP 지지되면 고가 재시도, 못 지키면 관망"},
             {"k": "약세", "t": f"전일 종가 {P(ref)} 이탈·거래량 감소 시 매물 출회 → 손절 참고 {P(it.get('stop'))}"}]
    if r:
        lines.append({"k": "리스크", "t": " · ".join(x[0] for x in r[:3]) + " — 급등 시 물량 출회 주의"})
    return lines


def jb_eval(it, f, df, jb) -> dict | None:
    sn = it.get("snap") or {}
    if not sn.get("reg_last") or not sn.get("prev") or df is None or not sn.get("sdate"):
        return None
    day = date.fromisoformat(sn["sdate"])
    past = df[[ts.date() < day for ts in df.index]]
    if len(past) < 21:
        return None
    avg20 = float(past["Volume"].iloc[-20:].mean())
    last, prev = sn["reg_last"], sn["prev"]
    hi, lo = sn.get("reg_hi") or last, sn.get("reg_lo") or last
    relvol = sn.get("reg_vol", 0) / avg20 if avg20 else float("nan")
    chg = (last / prev - 1) * 100
    cp = (last - lo) / (hi - lo) if hi > lo else 0.5
    ks = jbkeys(relvol, chg, last, cp)
    p10, n = jb.p(ks)
    st = jb.stats(ks)
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return {"day": f"{day.month}/{day.day}", "next": f"{nxt.month}/{nxt.day}({WEEK[nxt.weekday()]})",
            "price": last, "chg": round(chg, 1), "relvol": round(relvol, 1) if relvol == relvol else None,
            "cp": round(cp, 2), "p10": None if p10 is None else round(p10 * 100, 1), "n": n,
            "base": None if jb.base is None else round(jb.base * 100, 1), "st": st,
            "done": bool(sn.get("reg_done")), "asof": sn.get("reg_time")}


def jb_scenario(it, j, d, r, df) -> list[dict]:
    st = j.get("st") or {}
    buy = j["price"]
    stop = buy * 0.92
    res = resistances(df, buy)
    t1 = res[0] if res else buy * (1 + JB_HIT)
    ev = [x[2] for x in d if not x[0].startswith(("저유통", "전일 강세"))][:2]
    lines = [{"k": "근거", "t": f"오늘 {j['chg']:+.1f}% · 거래량 {j['relvol'] or '-'}배 · 종가 위치 {j['cp'] * 100:.0f}%"
                               + (" · " + " · ".join(ev) if ev else "")},
             {"k": "매수", "t": f"마감 직전 {P(buy)} 부근 분할 (현재가 기준, {j['asof'] or '-'} KST)"},
             {"k": "익일", "t": f"갭상승 확률 {st.get('up', '-')}% · 평균 시가 갭 {st.get('gap', 0):+.1f}% · "
                              f"+{int(JB_HIT * 100)}% 터치 {j['p10']}% (평균 {j['base']}%)"},
             {"k": "강세", "t": f"프리장 갭상승 후 {P(t1)}({pct(t1, buy)}) 돌파 시 추가 상승"
                              + (f", 터지면 평균 고가 +{st['hit_hi']:.0f}%" if st.get("hit_hi") else "")},
             {"k": "약세", "t": f"시가가 매수가 아래면 미련 없이 정리 · 손절 참고 {P(stop)}(-8%)"}]
    if r:
        lines.append({"k": "리스크", "t": " · ".join(x[0] for x in r[:3]) + " — 장 마감 후 증자 공시 위험"})
    return lines


def build_candidates(items, prices, frames, jb):
    by_t = {it["t"]: it for it in items}
    # 1차 점수 (외부 조회 없이) → 상위만 뉴스·공매도 조회
    pre_s = []
    for it in items:
        p = it.get("prob") or {}
        if p.get("reg") is None:
            continue
        d, r = drivers_for(it, {}, False)
        sc = min(55, p["reg"] * 2.5) + min(15, (p.get("pre") or 0) * 0.5) + sum(x[1] for x in d) - min(25, sum(x[1] for x in r))
        pre_s.append((sc, it["t"]))
    jbs = []
    for it in items:
        j = jb_eval(it, frames.get(it["t"]), prices.get(it["t"]), jb)
        if j and j["p10"] is not None:
            it["jb"] = j
            jbs.append((j["p10"] * 2 + (j["st"].get("up", 50) - 50) * 0.5, it["t"]))
    pick = [t for _, t in sorted(pre_s, reverse=True)[:15]] + [t for _, t in sorted(jbs, reverse=True)[:12]]
    ex = {}
    for t in dict.fromkeys(pick):
        try:
            ex[t] = enrich(t)
        except Exception as e:
            log.warning("%s 뉴스·공매도 조회 실패: %s", t, e)
            ex[t] = {}
    surge, jbc = [], []
    for t in dict.fromkeys(t for _, t in sorted(pre_s, reverse=True)[:15]):
        it = by_t[t]
        it["ex"] = ex.get(t, {})
        d, r = drivers_for(it, it["ex"], False)
        p = it["prob"]
        score = min(55, p["reg"] * 2.5) + min(15, (p.get("pre") or 0) * 0.5) + sum(x[1] for x in d) - min(25, sum(x[1] for x in r))
        surge.append({"t": t, "name": it["name"], "score": int(max(0, min(100, score))),
                      "type": d[0][0] if d else "확률형", "drivers": [x[0] for x in d], "risks": [x[0] for x in r],
                      "prob": p, "price": it["price"], "chg": it["chg"],
                      "scenario": surge_scenario(it, d, r, prices.get(t)), "news": it["ex"].get("news", []),
                      "si": it["ex"].get("si"), "dtc": it["ex"].get("dtc"), "float": it.get("float")})
    surge.sort(key=lambda x: -x["score"])
    for _, t in sorted(jbs, reverse=True)[:12]:
        it, j = by_t[t], by_t[t]["jb"]
        it["ex"] = ex.get(t, {})
        d, r = drivers_for(it, it["ex"], True)
        if j["chg"] >= 10 and j["cp"] >= 0.75:
            typ = "강세 마감형"
        elif j["chg"] <= -10:
            typ = "과매도 반등형"
        elif it["ex"].get("pos"):
            typ = "재료 대기형"
        else:
            typ = d[0][0] if d else "수급형"
        st = j["st"] or {}
        score = j["p10"] * 2 + (st.get("up", 50) - 50) * 0.6 + sum(x[1] for x in d) * 0.5 - min(25, sum(x[1] for x in r))
        jbc.append({"t": t, "name": it["name"], "score": int(max(0, min(100, score))), "type": typ,
                    "drivers": [x[0] for x in d], "risks": [x[0] for x in r], "jb": j,
                    "scenario": jb_scenario(it, j, d, r, prices.get(t)), "news": it["ex"].get("news", [])})
    jbc.sort(key=lambda x: -x["score"])
    MIN_SCORE = 25                       # 이 점수 미만은 후보에서 제외
    return [c for c in surge if c["score"] >= MIN_SCORE][:SURGE_TOP], [c for c in jbc if c["score"] >= MIN_SCORE][:JB_TOP]


# ───────────────── 5-4. 예정 호재 캘린더 + 알림 성적표 ─────────────────
CAL_TOP = 60
CAL_DAYS = 14
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
PDUFA_RE = re.compile(r"PDUFA[^.]{0,120}?((?:%s)\s+\d{1,2},\s+\d{4})" % MONTHS, re.I)
BIO_RE = re.compile(r"therapeut|pharma|bio|medic|oncolog|genetic|vaccin|clinical", re.I)


def _events_for(t: str, name: str) -> list[dict]:
    out = []
    tk = yf.Ticker(t)
    try:
        cal = tk.calendar or {}
        eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        for d in (eds or [])[:1]:
            d = d if isinstance(d, date) else pd.Timestamp(d).date()
            out.append({"date": d.isoformat(), "ev": "실적 발표", "note": "서프라이즈 시 급등·급락"})
    except Exception:
        pass
    if BIO_RE.search(name or ""):
        try:
            for it in (tk.news or [])[:20]:
                c = it.get("content") or it
                text = (c.get("title") or "") + " " + (c.get("summary") or "")
                for m in PDUFA_RE.finditer(text):
                    try:
                        d = datetime.strptime(m.group(1).replace("  ", " "), "%B %d, %Y").date()
                        out.append({"date": d.isoformat(), "ev": "FDA 심사일(PDUFA)", "note": "승인 시 대형 급등 재료"})
                    except ValueError:
                        pass
        except Exception:
            pass
    return out


def build_calendar(items, surge, jbc) -> list[dict]:
    today = datetime.now(ET).date()
    cache = cache_load("cal")
    picks = [it for it in items[:CAL_TOP]]
    extra = {c["t"] for c in (surge or []) + (jbc or [])} | {it["t"] for it in items if it.get("watch")}
    picks += [it for it in items if it["t"] in extra and it not in picks]
    ev = []
    for it in picks:
        t = it["t"]
        if t not in cache:
            try:
                cache[t] = _events_for(t, it["name"])
            except Exception:
                cache[t] = []
        for e in cache[t]:
            ev.append({"t": t, "name": it["name"], **e})
    for it in items:                              # 공시 기반 기한 (모든 종목)
        for d in it.get("ddays", []):
            ev.append({"t": it["t"], "name": it["name"], "date": d["date"], "ev": d["label"],
                       "note": "기한 전 부양성 호재·병합 발표 잦음" if "회복" in d["label"] else "공시 마감 전후 증자·실적 발표"})
    seen, out = set(), []
    for e in sorted(ev, key=lambda e: (e["date"], e["t"])):
        try:
            d = date.fromisoformat(e["date"])
        except ValueError:
            continue
        k = (e["t"], e["ev"], e["date"])
        if k in seen or not (0 <= (d - today).days <= CAL_DAYS):
            continue
        seen.add(k)
        e["d"] = (d - today).days
        e["label"] = f"{d.month}/{d.day}({WEEK[d.weekday()]})"
        out.append(e)
    return out


def scorecard() -> dict:
    try:
        st = json.loads((ROOT / "state.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    logs = st.get("alog", [])
    done = [e for e in logs if e.get("max_gain") is not None]

    def agg(xs):
        g = sorted(e["max_gain"] for e in xs)
        if not g:
            return None
        dd = [e["min_gain"] for e in xs if e.get("min_gain") is not None]
        return {"n": len(g), "hit10": round(sum(x >= 10 for x in g) / len(g) * 100), "hit20": round(sum(x >= 20 for x in g) / len(g) * 100),
                "avg": round(sum(g) / len(g), 1), "med": g[len(g) // 2], "dd": round(sum(dd) / len(dd), 1) if dd else None}
    rows = []
    for k in sorted({e["kind"] for e in done}):
        a = agg([e for e in done if e["kind"] == k])
        if a:
            rows.append({"k": k, **a})
    for g in ("S", "A", "B"):
        a = agg([e for e in done if e["kind"] == "촉매" and e.get("grade") == g])
        if a:
            rows.append({"k": f"촉매 {g}등급", **a})
    a = agg([e for e in done if e["kind"] == "점화" and e.get("combo")])
    if a:
        rows.append({"k": "점화(재료+수급)", **a})
    recent = []
    for e in reversed(logs[-40:]):
        recent.append({"t": e["t"], "k": e["kind"] + (f" {e['grade']}" if e.get("grade") else ""),
                       "at": datetime.fromisoformat(e["at"]).astimezone(KST).strftime("%m/%d %H:%M"),
                       "price": e["price"], "max": e.get("max_gain"), "min": e.get("min_gain")})
    return {"rows": rows, "recent": recent, "pending": sum(1 for e in logs if e.get("max_gain") is None and not e.get("checked"))}


# ───────────────── 6. AI 종합판단 (선택) ─────────────────
_groq = {"model": None, "used": 0}


def _groq_model() -> str | None:
    if _groq["model"]:
        return _groq["model"]
    if GROQ_MODEL:
        _groq["model"] = GROQ_MODEL
        return GROQ_MODEL
    try:
        ids = [m["id"] for m in http.get("https://api.groq.com/openai/v1/models", timeout=20,
                                         headers={"Authorization": f"Bearer {GROQ_KEY}"}).json().get("data", [])]
        for want in GROQ_PREFER:
            hit = next((i for i in ids if want in i and "guard" not in i), None)
            if hit:
                _groq["model"] = hit
                log.info("Groq 모델: %s", hit)
                return hit
    except Exception as e:
        log.warning("Groq 모델 목록 실패: %s", e)
    return None


def llm(prompt: str, max_tokens: int = 600) -> str:
    """Anthropic 키가 있으면 Claude, 없으면 Groq(무료). 둘 다 없으면 빈 문자열"""
    if ANTHROPIC_KEY:
        try:
            r = http.post("https://api.anthropic.com/v1/messages", timeout=90,
                          headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": AI_MODEL, "max_tokens": max_tokens,
                                "messages": [{"role": "user", "content": prompt}]})
            return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()
        except Exception as e:
            log.warning("Claude 실패: %s", e)
            return ""
    if GROQ_KEY and _groq["used"] < AI_PER_RUN:
        model = _groq_model()
        if not model:
            return ""
        _groq["used"] += 1
        for attempt in range(3):
            try:
                r = http.post("https://api.groq.com/openai/v1/chat/completions", timeout=90,
                              headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                              json={"model": model, "max_tokens": max_tokens, "temperature": 0.3,
                                    "messages": [{"role": "system", "content": "너는 미국 소형주 공시 분석가다. 반드시 한국어로만 답한다."},
                                                 {"role": "user", "content": prompt}]})
                if r.status_code == 429:
                    time.sleep(min(30, float(r.headers.get("retry-after", 10))))
                    continue
                txt = r.json()["choices"][0]["message"]["content"].strip()
                time.sleep(2.2)                    # 분당 30회 한도
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except Exception as e:
                log.warning("Groq 실패: %s", e)
                time.sleep(3)
    return ""


def ai_summary(item: dict) -> str:
    if not (ANTHROPIC_KEY or GROQ_KEY):
        return ""
    facts = {k: item[k] for k in ("t", "name", "price", "strength", "runway", "tags", "ddays", "stage")}
    facts["filings"] = [f"{f['date']} {f['form']} {f['items']}" for f in item["filings"][:15]]
    facts["today"] = [x["k"] + ": " + x["t"] for x in item["snap"]["signals"]]
    prompt = ("다음은 미국 소형주의 SEC 공시·재무·오늘 시세 요약이다. 회사가 주가를 부양해야 할 동기(상장유지, 현금조달, "
              "물량출회, 병합)와 위험(희석, 출회)을 한국어 3문장 이내로 판단하라. 확실하지 않은 건 '확인 필요'로 표시하고 "
              "매수 권유는 하지 마라.\n" + json.dumps(facts, ensure_ascii=False))
    c = cache_load("aisum")
    key = item["t"] + ":" + (item["filings"][0]["date"] if item.get("filings") else "")
    if key not in c:
        c[key] = llm(prompt, 400)
    return c[key]


# ───────────────── 6-2. 하루 캐시 (30분 갱신 시 큰 데이터 재다운로드 방지) ─────────────────
_CACHE: dict = {}
CALENDAR: list = []
FS_ALL: dict = {}
PRICES: dict = {}
REPORTS: list = []
REPORT_MAX = 90
FACT_KEYS = ["ClassOfWarrantOrRightOutstanding", "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1",
             "ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights",
             "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber",
             "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingWeightedAverageExercisePrice",
             "ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber",
             "PreferredStockSharesOutstanding", "ConvertibleNotesPayableCurrent", "ConvertibleNotesPayable",
             "ConvertibleDebtCurrent", "ConvertibleDebtNoncurrent", "Revenues",
             "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss"]


def facts_cached(cik) -> dict:
    """SEC companyfacts 중 필요한 항목만 하루 캐시"""
    if not cik:
        return {}
    c = cache_load("facts")
    if cik not in c:
        d = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json") or {}
        g = d.get("facts", {}).get("us-gaap", {})
        c[cik] = {"facts": {"us-gaap": {k: {"units": g[k].get("units", {})} for k in FACT_KEYS if k in g}}}
    return c[cik]


def profile_cached(t) -> dict:
    c = cache_load("prof")
    p = c.get(t)
    if p and p.get("summary_ko") and not HANGUL.search(p["summary_ko"]):
        p["summary_ko"] = translate_ko(p["summary_ko"])        # 지난번 번역 실패분 재시도
    if t not in c:
        try:
            c[t] = report.profile(t, translate_ko)
        except Exception as e:
            log.warning("%s 프로필 실패: %s", t, e)
            c[t] = {"inst": [], "insiders": []}
    return c[t]


def report_ai(rep: dict) -> str:
    if not (ANTHROPIC_KEY or GROQ_KEY):
        return ""
    c = cache_load("rai")
    key = rep["t"] + ":" + (rep["timeline"][0]["date"] if rep["timeline"] else "")
    if key in c:
        return c[key]
    facts = {k: rep.get(k) for k in ("t", "name", "price", "mcap", "shares", "float", "growth", "spike", "baby",
                                     "raises", "runway", "purposes", "events", "insiders", "inst")}
    facts["dil"] = rep["dil"]["rows"]
    facts["filings"] = [x["t"] for x in rep["timeline"][:15]]
    prompt = ("미국 소형주 분석가로서 아래 SEC·시세 데이터만 근거로 '종합 판단'을 한국어 4~6문장으로 써라. "
              "현금 조달·상장 유지·물량 출회 중 무엇이 핵심 목적인지, 펌핑 후 조달(희석) 시나리오, 확인이 필요한 공시를 "
              "구체적 날짜·금액·가격으로 적어라. 데이터에 없는 사실은 지어내지 말고 '확인 필요'라고 써라. 매수 권유 금지.\n"
              + json.dumps(_clean(facts), ensure_ascii=False, default=str)[:12000])
    txt = llm(prompt, 700)
    if txt:
        c[key] = txt                               # 실패하면 저장 안 함 → 다음 실행에 재시도
    return txt


def write_reports(items, surge, jbc) -> list[str]:
    by_t = {it["t"]: it for it in items}
    pick = [it["t"] for it in items[:60]] + [it["t"] for it in items if it.get("watch")]
    pick += [c["t"] for c in (surge or []) + (jbc or [])] + [d["t"] for d in DILUTION[:20]]
    pick = [t for t in dict.fromkeys(pick) if t in by_t][:REPORT_MAX]
    out_dir = OUT.parent / "r"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for t in pick:
        it = by_t[t]
        try:
            rep = report.build(it, PRICES.get(t), FS_ALL.get(t, []), facts_cached(it.get("cik")),
                               profile_cached(t), CALENDAR, classify_filing)
            ai = report_ai(rep)
            if ai:
                rep["verdict"], rep["ai"] = ai, True
            (out_dir / f"{t}.json").write_text(json.dumps(_clean(rep), ensure_ascii=False, default=str),
                                               encoding="utf-8")
            done.append(t)
        except Exception as e:
            log.warning("%s 리포트 실패: %s", t, e)
    tpl = ROOT / "report_template.html"
    if tpl.exists():
        (OUT.parent / "r.html").write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
    save_caches()
    log.info("상세 리포트 %d개", len(done))
    return done
FEED: list = []
DILUTION: list = []


def _cday() -> str:
    return datetime.now(ET).date().isoformat()


def cache_load(name: str):
    if name not in _CACHE:
        try:
            obj = pickle.loads((CACHE / f"{name}.pkl").read_bytes())
            _CACHE[name] = obj["data"] if obj.get("day") == _cday() else {}
        except Exception:
            _CACHE[name] = {}
    return _CACHE[name]


def save_caches():
    CACHE.mkdir(exist_ok=True)
    for name, data in _CACHE.items():
        try:
            (CACHE / f"{name}.pkl").write_bytes(pickle.dumps({"day": _cday(), "data": data}))
        except Exception as e:
            log.warning("캐시 저장 실패 %s: %s", name, e)


def cache_get(name, fn):
    c = cache_load(name)
    if not c:
        c.update(fn() or {})
    return c


def daily_cached(tickers):
    c = cache_load("daily")
    miss = [t for t in tickers if t not in c]
    if miss:
        c.update(daily(miss))
        for t in miss:
            c.setdefault(t, None)
    log.info("일봉 캐시 %d · 새로 받음 %d", len(tickers) - len(miss), len(miss))
    return {t: c[t] for t in tickers if c.get(t) is not None}


def premarket_highs_cached(tickers):
    c = cache_load("prehi")
    miss = [t for t in tickers if t not in c]
    if miss:
        c.update(premarket_highs(miss))
        for t in miss:
            c.setdefault(t, None)
    return {t: c[t] for t in tickers if c.get(t) is not None}


def sec_cached(cik):
    if not cik:
        return [], {}
    c = cache_load("sec2")
    if cik not in c:
        c[cik] = (filings(cik), financials(cik))
    return c[cik]


# ───────────────── 7. 조립 ─────────────────
FILING_FORMS = ("8-K", "8-K/A", "424B5", "424B4", "424B3", "S-1", "S-1/A", "S-3", "S-3/A", "F-1", "F-3",
                "EFFECT", "DEF 14A", "PRE 14A", "NT 10-Q", "NT 10-K", "10-Q", "10-K", "6-K",
                "SC 13D", "SC 13D/A", "SC 13G")

# ── 공시 호재·악재 분류 ──
ITEM_KO = {"1.01": ("호재", "주요 계약 체결"), "2.01": ("호재", "인수·합병 완료"), "1.02": ("악재", "계약 해지"),
           "1.03": ("악재", "파산"), "2.02": ("중립", "실적 발표"), "2.04": ("악재", "채무 조기상환 사유"),
           "3.01": ("악재", "상장 규정 미달 통지"), "3.02": ("악재", "미등록 주식 발행(희석)"),
           "3.03": ("주의", "주주 권리 변경"), "4.01": ("주의", "감사인 교체"), "4.02": ("악재", "과거 재무제표 신뢰 불가"),
           "5.02": ("주의", "임원 변동"), "5.03": ("주의", "정관 변경(병합 가능)"), "5.07": ("중립", "주총 결과"),
           "7.01": ("중립", "IR·보도자료"), "8.01": ("중립", "기타 중요 공시")}
FORM_SIDE = {"424B5": ("악재", "ATM·공모 설명서 (희석)"), "424B4": ("악재", "공모가 확정 (희석)"),
             "424B3": ("악재", "재판매 등록 (물량 출회)"), "S-1": ("악재", "증권신고서 (공모 준비)"),
             "S-1/A": ("악재", "증권신고서 정정 (공모 임박)"), "F-1": ("악재", "증권신고서 (해외기업 공모)"),
             "S-3": ("주의", "선반등록 Shelf (언제든 증자 가능)"), "S-3/A": ("주의", "Shelf 정정"),
             "F-3": ("주의", "선반등록 Shelf (해외기업)"), "EFFECT": ("주의", "신고서 효력 발생 — 곧 발행 가능"),
             "NT 10-Q": ("악재", "분기보고 지연"), "NT 10-K": ("악재", "연간보고 지연"),
             "SC 13D": ("호재", "5% 이상 지분 취득 (경영 참여 목적)"), "SC 13D/A": ("주의", "대주주 지분 변동"),
             "SC 13G": ("중립", "5% 이상 보유 보고"), "PRE 14A": ("주의", "주총 안건 (병합·증자 승인 가능)"),
             "DEF 14A": ("중립", "주총 위임장"), "10-Q": ("중립", "분기보고서"), "10-K": ("중립", "연간보고서"),
             "6-K": ("중립", "해외기업 수시공시")}
SIDE_RANK = {"악재": 3, "주의": 2, "호재": 1, "중립": 0}


def classify_filing(f: dict) -> tuple[str, str]:
    form = f["form"]
    if form.startswith("8-K"):
        its = [i.strip() for i in f.get("items", "").split(",") if i.strip() and i.strip() != "9.01"]
        labs = [ITEM_KO.get(i, ("중립", f"Item {i}")) for i in its]
        if "1.01" in its and "3.02" in its:
            return "악재", "증자 계약 체결 (주식·워런트 발행)"
        if not labs:
            return "중립", "수시공시"
        side = max((x[0] for x in labs), key=lambda v: SIDE_RANK[v] if v != "호재" else 1.5)
        return side, " · ".join(dict.fromkeys(x[1] for x in labs))
    return FORM_SIDE.get(form, ("중립", form))


FEED_DAYS = 10


def build_feed(items) -> list[dict]:
    cutoff = (datetime.now(KST).date() - timedelta(days=FEED_DAYS)).isoformat()
    out = []
    for it in items:
        for f in it.get("filings", []):
            if f["date"] < cutoff:
                continue
            side, label = classify_filing(f)
            if side == "중립" and f["form"] in ("10-Q", "10-K", "DEF 14A", "6-K"):
                continue
            out.append({"t": it["t"], "name": it["name"], "date": f["date"], "form": f["form"], "side": side,
                        "label": label, "url": f["url"], "price": it["price"], "chg": it.get("chg"), "mcap": it.get("mcap")})
    out.sort(key=lambda x: (x["date"], SIDE_RANK[x["side"]]), reverse=True)
    return out[:200]


def build_dilution(items) -> list[dict]:
    today = datetime.now(KST).date()
    out = []
    for it in items:
        fs = it.get("filings", [])

        def recent(forms, days, item=None):
            for f in fs:
                if f["form"] in forms and (today - date.fromisoformat(f["date"])).days <= days \
                        and (item is None or item in f.get("items", "")):
                    return f
            return None
        flags, score, last = [], 0, None
        for forms, days, item, pts, lab in [
            (("424B5", "424B4"), 120, None, 3, "ATM·공모 진행"),
            (("S-1", "S-1/A", "F-1"), 120, None, 3, "공모 준비(S-1)"),
            (("8-K", "8-K/A"), 180, "3.02", 2, "미등록 발행"),
            (("424B3",), 180, None, 2, "재판매 물량"),
            (("EFFECT",), 60, None, 1, "신고서 효력"),
            (("S-3", "S-3/A", "F-3"), 3 * 365, None, 1, "Shelf 보유"),
        ]:
            f = recent(forms, days, item)
            if f:
                flags.append(lab)
                score += pts
                last = max(last or f["date"], f["date"])
        rw = it.get("runway")
        if rw is not None and rw < 6:
            score += 2; flags.append(f"현금 {rw:.1f}개월")
        elif rw is not None and rw < 12:
            score += 1; flags.append(f"현금 {rw:.0f}개월")
        sh = it.get("sh") or {}
        g = sh.get("growth")
        if g is not None:
            if g >= 100:
                score += 3
            elif g >= 30:
                score += 2
            elif g >= 10:
                score += 1
        if any(t["name"] in ("병합", "병합 이력") for t in it["tags"]):
            score += 1; flags.append("병합")
        if score < 2:
            continue
        out.append({"t": it["t"], "name": it["name"], "price": it["price"], "chg": it.get("chg"),
                    "score": score, "risk": "높음" if score >= 6 else "중간" if score >= 4 else "낮음",
                    "flags": flags, "growth": g, "sh_now": sh.get("now"), "sh_prev_date": sh.get("prev_date"),
                    "sh_date": sh.get("now_date"), "runway": rw, "last": last, "float": it.get("float")})
    out.sort(key=lambda x: (-x["score"], -(x["growth"] or 0)))
    return out[:80]


def analyze() -> list[dict]:
    today = datetime.now(KST).date()
    uni = universe()
    if not uni:
        return [], [], []
    tickers = [u["t"] for u in uni]
    prices = daily_cached(tickers)
    PRICES.update(prices)
    mins = intraday(tickers)
    log.info("일봉 %d · 1분봉 %d", len(prices), len(mins))
    frames = {t: tech_frame(df) for t, df in prices.items()}
    stats = similar_stats(frames)
    target = target_session()
    reg_m, pre_m, jb_m = build_models(frames, premarket_highs_cached(tickers), target)
    ciks = cache_get("cik", cik_map)
    items = []
    for u in uni:
        t = u["t"]
        try:
            df = prices.get(t)
            snap = snapshot(u, df, mins.get(t))
            price = snap["price"] or u["price"] or 0
            cik = ciks.get(t)
            fs, fin = sec_cached(cik)
            FS_ALL[t] = fs
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
                "equity": fin.get("equity"), "sh": fin.get("sh"), "tags": tags, "ddays": ddays,
                "stage": stg, "stage_name": STAGE_NAME[stg],
                "checks": {k: bool(frames[t][k].iloc[-1]) for k in ("ma20", "lows", "upvol", "liq", "cool")}
                if t in frames else {},
                "similar": st, "target": price * mult if mult and mult > 1.02 else None, "stop": stop, "total": total,
                "snap": snap, "prob": predict(frames.get(t), reg_m, pre_m, target, snap),
                "filings": [dict(f, ko=FORM_KO.get(f"{f['form']}:{f['items'][:4]}" if f["form"] == "8-K"
                                                   else f["form"], "")) for f in fs if f["form"] in FILING_FORMS][:20],
                "ai": "",
            })
        except Exception as e:
            log.warning("%s 분석 실패: %s", t, e)
    items.sort(key=lambda x: x["total"], reverse=True)
    for it in items[:AI_TOP]:
        it["ai"] = ai_summary(it)
    surge, jbc = build_candidates(items, prices, frames, jb_m)
    global CALENDAR, FEED, DILUTION
    for name, fn in (("캘린더", lambda: build_calendar(items, surge, jbc)), ("공시", lambda: build_feed(items)),
                     ("희석", lambda: build_dilution(items))):
        try:
            val = fn()
            if name == "캘린더":
                CALENDAR = val
            elif name == "공시":
                FEED = val
            else:
                DILUTION = val
        except Exception as e:
            log.warning("%s 실패: %s", name, e)
    save_caches()
    return items, surge, jbc


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


def write_page(items: list[dict], surge=None, jbc=None) -> None:
    tpl = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    payload = _clean({"updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"), "items": items,
                      "forward": FORWARD_DAYS, "surge": surge or [], "jbc": jbc or [],
                      "cal": CALENDAR, "dil": DILUTION, "reports": REPORTS,
                      "rules": {"reg": int(SURGE_REG * 100), "pre": int(SURGE_PRE * 100), "jb": int(JB_HIT * 100)}})
    data = json.dumps(payload, ensure_ascii=False, default=lambda o: None).replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tpl.replace("/*__DATA__*/null", data), encoding="utf-8")
    (OUT.parent / ".nojekyll").write_text("")
    log.info("대시보드 저장: %s (%d종목)", OUT, len(items))


TELEGRAM_HOURS_KST = {21, 1, 5}   # 매시간 갱신해도 텔레그램 요약은 이 시간대에만 (수동 실행은 항상)


def telegram(items: list[dict], surge=None, jbc=None) -> None:
    if not TOKEN:
        return
    manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    slot = datetime.now(KST).strftime("%Y-%m-%d %H")
    sent = cache_load("tg")
    if not manual and (datetime.now(KST).hour not in TELEGRAM_HOURS_KST or sent.get("slot") == slot):
        log.info("텔레그램 요약 시간 아님 — 대시보드만 갱신")
        return
    sent["slot"] = slot
    save_caches()
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
    pr = sorted([it for it in items if it.get("prob") and it["prob"].get("reg") is not None],
                key=lambda it: it["prob"]["reg"], reverse=True)[:5]
    if pr:
        p0 = pr[0]["prob"]
        prow = [f"{it['t']:<5} {it['prob']['pre'] or 0:>5.1f}% {it['prob']['reg']:>5.1f}%" for it in pr]
        body += ("\n\n" + f"{p0['date']} 급등확률 TOP5\n티커   프리+{p0['pre_th']}% 본장+{p0['reg_th']}%\n"
                 + "─" * 26 + "\n" + "\n".join(prow) + f"\n평균   {p0['pre_base'] or 0:>5.1f}% {p0['reg_base'] or 0:>5.1f}%")
    if surge:
        body += "\n\n🚀 금일 급등 후보\n" + "\n".join(
            f"{c['t']:<5} {c['score']:>3}점 본장{c['prob']['reg']:>5.1f}% {c['type']}" for c in surge[:3])
    if jbc:
        body += "\n\n🎯 종배 예상 (" + jbc[0]["jb"]["day"] + " 종가)\n" + "\n".join(
            f"{c['t']:<5} {c['score']:>3}점 익일+{int(JB_HIT * 100)}% {c['jb']['p10']:>5.1f}% {c['type']}" for c in jbc[:3])
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
    items, surge, jbc = analyze()
    if not items:
        log.error("분석 결과 없음")
        return
    global REPORTS
    try:
        REPORTS = write_reports(items, surge, jbc)
    except Exception as e:
        log.exception("리포트 생성 실패: %s", e)
    write_page(items, surge, jbc)
    telegram(items, surge, jbc)


if __name__ == "__main__":
    main()
