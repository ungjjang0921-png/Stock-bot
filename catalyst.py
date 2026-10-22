"""
촉매(호재·공시) 실시간 감지 + 거래량 점화 감지 + 알림 성적표
════════════════════════════════════════════
1) 촉매 감지  : 보도자료(GlobeNewswire·PR Newswire·Business Wire) + SEC 실시간 공시(8-K·6-K·13D·증자서류)
               → 미국 상장 저유통 소형주에 호재가 뜨는 순간 등급(S/A/B)과 함께 알림
               → 증자·희석 공시는 '설거지 경고'로 따로 알림
2) 점화 감지  : 미국 전 종목을 훑어 '직전 확인 대비' 거래량·주가가 막 터지기 시작한 초입 종목 알림
               → 최근 3시간 안에 촉매가 있었으면 '재료+수급 동시'로 최우선 표시
3) 성적표     : 모든 알림의 이후 최고가를 자동 채점 → 유형·등급별 적중률 (/score, 대시보드)

stock_alert_bot.py 가 불러서 씁니다. 따로 실행하지 않아요.
"""

import html
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

ETZ = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
log = logging.getLogger("catalyst")
http = requests.Session()
SEC_EMAIL = os.getenv("SEC_EMAIL", "").strip()
UA_WEB = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
UA_SEC = {"User-Agent": f"stock-alert-bot {SEC_EMAIL or 'contact@example.com'}"}

# ═════════════════════ 설정 ═════════════════════
FEEDS = [
    ("GlobeNewswire", "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/"
                      "GlobeNewswire%20-%20News%20about%20Public%20Companies"),
    ("PR Newswire", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    ("Business Wire", "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpRWQ=="),
]
SEC_FORMS = ["8-K", "6-K", "SC 13D", "424B5", "424B4", "424B3", "S-1", "S-1/A", "F-1", "S-3", "EFFECT",
             "NT 10-Q", "NT 10-K"]
SEC_DIGEST = True                # 소형주 SEC 공시(호재·악재·주의)를 1분 단위로 묶어 실시간 전송

# 공시 분류 (form · 8-K item → 호재/악재/주의)
D_ITEMS = {"1.01": ("호재", "주요 계약 체결"), "2.01": ("호재", "인수·합병 완료"), "1.02": ("악재", "계약 해지"),
           "1.03": ("악재", "파산"), "2.04": ("악재", "채무 조기상환 사유"), "3.01": ("악재", "상장 규정 미달 통지"),
           "3.02": ("악재", "미등록 주식 발행(희석)"), "4.02": ("악재", "과거 재무제표 신뢰 불가"),
           "3.03": ("주의", "주주 권리 변경"), "4.01": ("주의", "감사인 교체"), "5.03": ("주의", "정관 변경(병합 가능)")}
D_FORMS = {"424B5": ("악재", "ATM·공모 설명서 (희석)"), "424B4": ("악재", "공모가 확정 (희석)"),
           "424B3": ("악재", "재판매 등록 (물량 출회)"), "S-1": ("악재", "증권신고서 (공모 준비)"),
           "S-1/A": ("악재", "증권신고서 정정 (공모 임박)"), "F-1": ("악재", "증권신고서 (해외기업 공모)"),
           "S-3": ("주의", "선반등록 Shelf (언제든 증자 가능)"), "EFFECT": ("주의", "신고서 효력 발생 — 곧 발행 가능"),
           "NT 10-Q": ("악재", "분기보고 지연"), "NT 10-K": ("악재", "연간보고 지연"),
           "SC 13D": ("호재", "5% 이상 지분 취득 (경영 참여)")}


def sec_side(form: str, items: list[str]) -> tuple[str, str]:
    if form.startswith("8-K") or form == "6-K":
        its = [i for i in items if i != "9.01"]
        if "1.01" in its and "3.02" in its:
            return "악재", "증자 계약 체결 (주식·워런트 발행)"
        labs = [D_ITEMS[i] for i in its if i in D_ITEMS]
        if not labs:
            return "중립", "수시공시"
        rank = {"악재": 3, "주의": 2, "호재": 1}
        side = max((x[0] for x in labs), key=lambda v: rank[v])
        return side, " · ".join(dict.fromkeys(x[1] for x in labs))
    return D_FORMS.get(form, ("중립", form))
MAX_AGE_MIN = 45                 # 이보다 오래된 보도자료는 무시 (GitHub 지연 감안)
MAX_PRICE = 30.0
MAX_MCAP = 2e9
MAX_FLOAT = 60e6
MIN_GRADE = "B"                  # 이 등급 이상만 촉매 알림

IGN_MIN_CHG, IGN_MAX_CHG = 5.0, 60.0     # 점화: 이 등락률 구간(초입)만
IGN_MIN_STEP = 3.0                       # 직전 확인 대비 최소 상승 %
IGN_MIN_DOLLAR_STEP = 50_000             # 직전 확인 대비 최소 거래대금 증가($)
IGN_MAX_FLOAT = 50e6
IGN_PER_RUN = 5
EVAL_PER_RUN = 15
# ═══════════════════════════════════════════════

US_EX = {"NASDAQ", "NYSE", "NYSE AMERICAN", "NYSEAMERICAN", "AMEX", "NYSE MKT", "NYSEMKT"}
TICK_RE = re.compile(
    r"\b(NASDAQ|Nasdaq|NYSE American|NYSE MKT|NYSE|AMEX|NYSEAMERICAN)"
    r"(?:\s+(?:Capital|Global(?:\s+Select)?)\s+Market)?\s*:\s*\"?([A-Z]{1,5}(?:\.[A-Z])?)\b")

# (라벨, 가중치, 키워드) — 소형주 급등을 실제로 자주 만드는 재료 순
CATS = [
    ("피인수·합병", 10, ["to be acquired", "definitive agreement to be acquired", "merger agreement",
                       "acquire all outstanding", "per share in cash", "tender offer", "take-private"]),
    ("FDA 승인", 10, ["fda approval", "fda approves", "approved by the fda", "fda clearance", "510(k) clearance",
                    "marketing authorization", "ema approval"]),
    ("임상 성공", 9, ["positive topline", "met its primary endpoint", "met the primary endpoint",
                    "statistically significant", "positive results", "positive data", "positive interim"]),
    ("FDA 지정·혜택", 7, ["breakthrough therapy", "fast track", "orphan drug", "rare pediatric", "priority review"]),
    ("빅테크·빅파마 제휴", 8, []),     # BIG_NAMES 로 판정
    ("대형 계약·수주", 7, ["purchase order", "awarded", "award of", "contract with", "supply agreement",
                        "order from", "million contract", "million order", "multi-year agreement"]),
    ("정부·국방 계약", 7, ["department of defense", "u.s. army", "u.s. navy", "u.s. air force", "dod ",
                         "government contract", "nasa", "darpa", "pentagon"]),
    ("테마(AI·크립토·양자 등)", 6, ["artificial intelligence", " ai ", "ai-powered", "bitcoin", "crypto",
                               "ethereum", "treasury strategy", "digital asset", "blockchain", "quantum",
                               "drone", "nuclear", "robotics", "stablecoin"]),
    ("실적 서프라이즈", 5, ["record revenue", "record quarter", "raises guidance", "raised guidance",
                         "exceeded", "profitability", "first profitable"]),
    ("자사주·내부자 매수", 4, ["share repurchase", "buyback", "insider purchase", "purchased shares"]),
    ("전략적 대안 검토", 6, ["strategic alternatives", "strategic review"]),
    ("특허", 3, ["patent"]),
]
BIG_NAMES = ["nvidia", "microsoft", "amazon", "google", "alphabet", "apple ", "meta platforms", "openai",
             "tesla", "oracle", "pfizer", "merck", "novartis", "eli lilly", "johnson & johnson", "astrazeneca",
             "sanofi", "roche", "bristol myers", "boeing", "lockheed", "raytheon", "spacex", "walmart"]
NEG = [("공모·희석", ["public offering", "registered direct", "private placement", "at-the-market",
                     "pricing of", "priced", "warrant inducement", "equity line"]),
       ("주식병합", ["reverse stock split", "reverse split"]),
       ("상폐·파산", ["bankruptcy", "chapter 11", "delisting determination", "going concern"])]
SEC_ITEMS = {"1.01": ("계약 체결 공시", 6), "2.01": ("인수 완료 공시", 6), "1.02": ("계약 해지", -4),
             "3.01": ("상장 규정 미달 통지", 0), "3.02": ("미등록 주식 발행(희석)", -8),
             "5.03": ("정관 변경(병합 가능)", -3), "1.03": ("파산", -10)}
GRADE_ORDER = {"S": 3, "A": 2, "B": 1, "C": 0}


# ───────────────── 공통 ─────────────────
def _num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _p(v):
    return "-" if v is None else (f"${v:.4f}" if v < 1 else f"${v:.2f}")


def _big(v):
    if v is None:
        return "-"
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s or ""))).strip()


def session_fields(now_et: datetime):
    t = now_et.time()
    if dtime(4, 0) <= t < dtime(9, 30):
        return "premarket_close", "premarket_change", "premarket_volume", "프리장"
    if dtime(9, 30) <= t < dtime(16, 0):
        return "close", "change", "volume", "정규장"
    if dtime(16, 0) <= t < dtime(20, 0):
        return "postmarket_close", "postmarket_change", "postmarket_volume", "애프터장"
    return None


def tv_query(body: dict) -> list[dict]:
    try:
        data = http.post("https://scanner.tradingview.com/america/scan", json=body, headers=UA_WEB, timeout=20).json()
        cols = body["columns"]
        return [{"sym": r["s"], "t": r["s"].split(":")[-1].replace(".", "-"), **dict(zip(cols, r["d"]))}
                for r in data.get("data", [])]
    except Exception as e:
        log.warning("TradingView 조회 실패: %s", e)
        return []


QUOTE_COLS = ["description", "close", "change", "volume", "premarket_close", "premarket_change",
              "premarket_volume", "postmarket_close", "postmarket_change", "postmarket_volume",
              "average_volume_10d_calc", "market_cap_basic", "float_shares_outstanding", "exchange"]


def quotes(tickers: list[str]) -> dict[str, dict]:
    if not tickers:
        return {}
    syms = [f"{ex}:{t.replace('-', '.')}" for t in tickers for ex in ("NASDAQ", "NYSE", "AMEX")]
    rows = tv_query({"markets": ["america"], "symbols": {"query": {"types": []}, "tickers": syms},
                     "columns": QUOTE_COLS, "range": [0, len(syms)]})
    return {r["t"]: r for r in rows}


def live(q: dict, now_et: datetime) -> tuple[float | None, float | None, float | None]:
    """현재 세션 기준 (가격, 등락%, 거래량)"""
    f = session_fields(now_et) or ("close", "change", "volume", "")
    price = _num(q.get(f[0])) or _num(q.get("close"))
    chg = _num(q.get(f[1]))
    if f[3] == "애프터장" and chg is not None and q.get("change") is not None:
        chg = ((1 + _num(q["change"]) / 100) * (1 + chg / 100) - 1) * 100   # 전일 종가 대비로 환산
    vol = _num(q.get(f[2])) or _num(q.get("volume"))
    return price, chg, vol


# ───────────────── 1. 촉매 수집 ─────────────────
def fetch_feed(name: str, url: str) -> list[dict]:
    try:
        r = http.get(url, headers=UA_WEB, timeout=15)
        root = ElementTree.fromstring(r.content)
    except Exception as e:
        log.warning("%s 피드 실패: %s", name, e)
        return []
    out = []
    for it in root.iter("item"):
        try:
            title = (it.findtext("title") or "").strip()
            desc = _strip_html(it.findtext("description") or "")
            link = (it.findtext("link") or "").strip()
            gid = (it.findtext("guid") or link or title).strip()
            pub = it.findtext("pubDate")
            ts = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
            tickers = set()
            for c in it.findall("category"):
                txt = (c.text or "").strip()
                if ":" in txt:
                    ex, tk = txt.split(":", 1)
                    if ex.strip().upper() in US_EX and re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", tk.strip()):
                        tickers.add(tk.strip().replace(".", "-"))
            for m in TICK_RE.finditer(title + " " + desc[:1500]):
                tickers.add(m.group(2).replace(".", "-"))
            out.append({"id": f"{name}:{gid}", "src": name, "title": title, "desc": desc[:1200], "link": link,
                        "ts": ts, "tickers": sorted(tickers)})
        except Exception:
            continue
    return out


def fetch_sec(form: str) -> list[dict]:
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={requests.utils.quote(form)}"
           "&company=&dateb=&owner=include&start=0&count=100&output=atom")
    try:
        r = http.get(url, headers=UA_SEC, timeout=15)
        root = ElementTree.fromstring(r.content)
    except Exception as e:
        log.warning("SEC %s 피드 실패: %s", form, e)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        try:
            title = e.findtext("a:title", "", ns)
            m = re.search(r"\((\d{6,10})\)", title)
            if not m or "(Filer)" not in title and "(Subject)" not in title:
                continue
            summary = _strip_html(e.findtext("a:summary", "", ns))
            link_el = e.find("a:link", ns)
            upd = e.findtext("a:updated", "", ns)
            out.append({"id": "SEC:" + e.findtext("a:id", "", ns) + ":" + m.group(1), "src": "SEC", "form": form,
                        "cik": int(m.group(1)), "title": title, "desc": summary,
                        "link": link_el.get("href") if link_el is not None else "",
                        "ts": datetime.fromisoformat(upd).astimezone(timezone.utc) if upd else None,
                        "items": re.findall(r"Item (\d\.\d\d)", summary)})
        except Exception:
            continue
    return out


_cik_cache = {"day": None, "map": {}}


def cik_to_ticker() -> dict[int, str]:
    today = datetime.now(ETZ).date()
    if _cik_cache["day"] != today:
        try:
            data = http.get("https://www.sec.gov/files/company_tickers.json", headers=UA_SEC, timeout=20).json()
            _cik_cache["map"] = {int(v["cik_str"]): v["ticker"].upper().replace(".", "-") for v in data.values()}
            _cik_cache["day"] = today
        except Exception as e:
            log.warning("CIK 목록 실패: %s", e)
    return _cik_cache["map"]


def classify(text: str) -> tuple[list[tuple[str, int]], list[str]]:
    t = " " + text.lower() + " "
    hits = []
    for label, w, words in CATS:
        if label == "빅테크·빅파마 제휴":
            names = [n.strip() for n in BIG_NAMES if n in t]
            if names and any(k in t for k in ("partner", "collaborat", "agreement", "selected", "integrat",
                                              "supplier", "deploy", "with ")):
                hits.append((f"{label}({names[0].title()})", w))
            continue
        if any(k in t for k in words):
            hits.append((label, w))
    neg = [label for label, words in NEG if any(k in t for k in words)]
    hits.sort(key=lambda x: -x[1])
    return hits, neg


def grade_of(score: float) -> str:
    return "S" if score >= 16 else "A" if score >= 12 else "B" if score >= 8 else "C"


# ───────────────── 2. 성적표 ─────────────────
def log_alert(state: dict, kind: str, t: str, price: float | None, **kw) -> None:
    if not price:
        return
    state.setdefault("alog", []).append({
        "kind": kind, "t": t, "price": price, "at": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(ETZ).date().isoformat(), "max_gain": None, "min_gain": None, **kw})
    state["alog"] = state["alog"][-2000:]


def evaluate_log(state: dict) -> None:
    """지난 거래일 알림의 이후 최고가·최저가 채점 (1분봉 7일 제공 한도)"""
    try:
        import yfinance as yf
    except ImportError:
        return
    today = datetime.now(ETZ).date().isoformat()
    todo = [e for e in state.get("alog", []) if e.get("max_gain") is None and not e.get("checked")
            and e["date"] < today][:EVAL_PER_RUN]
    for e in todo:
        try:
            bars = yf.Ticker(e["t"]).history(period="7d", interval="1m", prepost=True)
            idx = bars.index.tz_convert(ETZ)
            at = datetime.fromisoformat(e["at"]).astimezone(ETZ)
            after = bars[(idx >= at) & (idx.date == at.date())]
            if len(after):
                e["max_gain"] = round((float(after["High"].max()) / e["price"] - 1) * 100, 1)
                e["min_gain"] = round((float(after["Low"].min()) / e["price"] - 1) * 100, 1)
            else:
                e["checked"] = True
        except Exception as ex:
            log.warning("%s 채점 실패: %s", e["t"], ex)
            e["checked"] = True


def stats(state: dict, kind: str | None = None, key: str | None = None, val=None) -> dict:
    xs = [e for e in state.get("alog", []) if e.get("max_gain") is not None
          and (kind is None or e["kind"] == kind) and (key is None or e.get(key) == val)]
    if not xs:
        return {"n": 0}
    g = sorted(e["max_gain"] for e in xs)
    return {"n": len(xs), "hit10": sum(x >= 10 for x in g) / len(g) * 100, "hit20": sum(x >= 20 for x in g) / len(g) * 100,
            "avg": sum(g) / len(g), "med": g[len(g) // 2],
            "dd": sum(e["min_gain"] for e in xs if e.get("min_gain") is not None) / len(xs)}


def stat_line(state, kind, key=None, val=None, min_n=10) -> str:
    s = stats(state, kind, key, val)
    if s["n"] < min_n:
        return f"적중률 집계중({s['n']}/{min_n})"
    return f"과거 {s['n']}건: +10% {s['hit10']:.0f}% · +20% {s['hit20']:.0f}% · 평균최고 +{s['avg']:.0f}%"


def score_text(state: dict) -> str:
    kinds = sorted({e["kind"] for e in state.get("alog", [])})
    rows = []
    for k in kinds:
        s = stats(state, k)
        if s["n"]:
            rows.append(f"{k:<6} {s['n']:>4} {s['hit10']:>5.0f}% {s['hit20']:>5.0f}% {s['avg']:>+6.0f}% {s['dd']:>+6.0f}%")
    for g in ("S", "A", "B"):
        s = stats(state, "촉매", "grade", g)
        if s["n"]:
            rows.append(f"촉매 {g}등급 {s['n']:>3} {s['hit10']:>5.0f}% {s['hit20']:>5.0f}% {s['avg']:>+6.0f}% {s['dd']:>+6.0f}%")
    pending = sum(1 for e in state.get("alog", []) if e.get("max_gain") is None and not e.get("checked"))
    if not rows:
        return f"📊 <b>알림 성적표</b>\n아직 채점된 알림이 없어요. (채점 대기 {pending}건 — 다음 거래일에 채점)"
    body = "유형      건수  +10%  +20%  평균최고 평균최저\n" + "─" * 40 + "\n" + "\n".join(rows)
    return (f"📊 <b>알림 성적표</b> (알림가 대비 그날 최고·최저)\n<pre>{html.escape(body)}</pre>"
            f"채점 대기 {pending}건")


# ───────────────── 3. 촉매 알림 ─────────────────
def scan_catalysts(state: dict, send, translate, watch: list[str]) -> None:
    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ETZ)
    seen = state.setdefault("cat_seen", [])
    seen_set = set(seen)
    fresh = []
    for name, url in FEEDS:
        for it in fetch_feed(name, url):
            if it["id"] in seen_set:
                continue
            seen.append(it["id"]); seen_set.add(it["id"])
            if it["tickers"] and it["ts"] and now - it["ts"] <= timedelta(minutes=MAX_AGE_MIN):
                fresh.append(it)
    cmap = None
    for form in SEC_FORMS:
        for it in fetch_sec(form):
            if it["id"] in seen_set:
                continue
            seen.append(it["id"]); seen_set.add(it["id"])
            if not it["ts"] or now - it["ts"] > timedelta(minutes=MAX_AGE_MIN):
                continue
            cmap = cmap if cmap is not None else cik_to_ticker()
            tk = cmap.get(it["cik"])
            if tk:
                it["tickers"] = [tk]
                fresh.append(it)
        time.sleep(0.15)
    state["cat_seen"] = seen[-1500:]
    if not fresh:
        return

    q = quotes(sorted({t for it in fresh for t in it["tickers"]}))
    recent = state.setdefault("cat_recent", {})
    for k in [k for k, v in recent.items() if now - datetime.fromisoformat(v["ts"]) > timedelta(hours=6)]:
        del recent[k]
    alerts, digest = [], []
    for it in fresh:
        for t in it["tickers"][:3]:
            qt = q.get(t)
            if not qt:
                continue
            price, chg, vol = live(qt, now_et)
            mcap, fl = _num(qt.get("market_cap_basic")), _num(qt.get("float_shares_outstanding"))
            small = (price or 99) <= MAX_PRICE and ((mcap or 0) <= MAX_MCAP or (fl or 0) <= MAX_FLOAT)
            if not small and t not in watch:
                continue
            if it["src"] == "SEC" and SEC_DIGEST:
                side, lab = sec_side(it["form"], it["items"])
                if side != "중립":
                    digest.append((side, t, it, price, chg, fl, lab))
            if it["src"] == "SEC":
                hits, neg = [], []
                for i in it["items"]:
                    lab, w = SEC_ITEMS.get(i, ("", 0))
                    if w > 0:
                        hits.append((lab, w))
                    elif w < 0:
                        neg.append(lab)
                if it["form"] == "SC 13D":
                    hits.append(("5% 이상 지분 취득(13D)", 6))
                if it["form"] in ("424B5", "424B4", "S-1", "F-1"):
                    neg.append(f"증자 서류({it['form']})")
                if "3.01" in it["items"]:
                    hits.append(("상장유지 압박(부양 동기)", 3))
                if t in recent and recent[t].get("src") != "SEC":
                    continue                       # 같은 재료의 보도자료 알림이 이미 나감
            else:
                hits, neg = classify(it["title"] + " " + it["desc"])
            score = sum(w for _, w in hits[:2])
            if fl:
                score += 4 if fl < 5e6 else 2 if fl < 15e6 else 0
            if price and price < 5:
                score += 1
            if it["src"] != "SEC" and t.upper() in it["title"].upper():
                score += 1
            grade = grade_of(score)
            dilute = any(n.startswith(("공모", "증자", "미등록")) for n in neg)
            if dilute and (chg or 0) >= 15 or (dilute and t in watch):
                alerts.append(("warn", t, it, qt, price, chg, vol, fl, mcap, hits, neg, "W", score))
            elif not neg and GRADE_ORDER[grade] >= GRADE_ORDER[MIN_GRADE]:
                alerts.append(("cat", t, it, qt, price, chg, vol, fl, mcap, hits, neg, grade, score))
    alerts.sort(key=lambda a: -a[12])
    alerted = {a[1] for a in alerts[:6]}
    rows = [d for d in digest if d[1] not in alerted]
    if rows:
        order = {"악재": 0, "주의": 1, "호재": 2}
        rows.sort(key=lambda d: order[d[0]])
        icon = {"악재": "🔴", "주의": "🟠", "호재": "🟢"}
        lines = []
        for side, t, it, price, chg, fl, lab in rows[:12]:
            ch = f" {chg:+.1f}%" if chg is not None else ""
            when = it["ts"].astimezone(KST).strftime("%H:%M")
            link = f' · <a href="{html.escape(it["link"])}">원문</a>' if it["link"] else ""
            lines.append(f"{icon[side]} <b>{t}</b> {side} · {html.escape(lab)} ({it['form']})\n"
                         f"    {_p(price)}{ch} · 유통 {_big(fl)} · {when}{link}")
        send(f"📑 <b>실시간 공시</b> ({len(rows)}건)\n" + "\n".join(lines))
    for kind, t, it, qt, price, chg, vol, fl, mcap, hits, neg, grade, score in alerts[:6]:
        age = int((now - it["ts"]).total_seconds() // 60)
        ko = translate(it["title"]) if it["src"] != "SEC" else it["title"]
        when = it["ts"].astimezone(ETZ).strftime("%H:%M ET")
        sess = (session_fields(now_et) or (0, 0, 0, "장외"))[3]
        mkt = (f"💰 {_p(price)} ({sess} {chg:+.1f}%)" if chg is not None else f"💰 {_p(price)}") + \
              f" · 거래량 {_big(vol)} · 유통 {_big(fl)} · 시총 ${_big(mcap)}"
        link = f'\n🔗 <a href="{html.escape(it["link"])}">원문</a>' if it["link"] else ""
        if kind == "warn":
            send(f"🚨 <b>설거지 경고</b> {t} — {', '.join(neg)}\n🕒 {when} ({age}분 전) · {it['src']}\n"
                 f"💬 {html.escape(ko)}\n{mkt}{link}\n⚠️ 급등 중 증자·희석 공시 = 물량 출회 위험")
            log_alert(state, "희석경고", t, price)
            continue
        import guard
        ok, bad, ginfo = guard.pre_check(t, price or 0, chg) if price else (True, [], {"upside": None})
        if not ok:
            continue                                     # 호재여도 설거지 위험이면 제외
        labels = " · ".join(h[0] for h in hits[:2])
        icon = {"S": "🔥🔥", "A": "🔥", "B": "📰"}[grade]
        send(f"{icon} <b>촉매 포착 [{grade}]</b> {t} · {html.escape(labels)}\n"
             f"🕒 {when} ({age}분 전) · {it['src']}\n💬 {html.escape(ko)}\n{mkt}\n"
             f"📊 {grade}등급 {stat_line(state, '촉매', 'grade', grade)}{link}\n"
             f"⚡ 거래량이 붙는지 확인 후 대응 — 재료만으로 오르지 않을 수 있어요")
        recent[t] = {"label": labels, "grade": grade, "ts": now.isoformat(), "title": ko[:80], "src": it["src"]}
        log_alert(state, "촉매", t, price, grade=grade, cat=hits[0][0] if hits else "")
        guard.track(state, t, price, "촉매")


# ───────────────── 4. 거래량 점화 감지 ─────────────────
def scan_ignition(state: dict, send) -> None:
    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ETZ)
    f = session_fields(now_et)
    if not f:
        return
    pf, cf, vf, sess = f
    rows = tv_query({
        "markets": ["america"],
        "filter": [{"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                   {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                   {"left": "close", "operation": "in_range", "right": [0.1, MAX_PRICE]},
                   {"left": cf, "operation": "greater", "right": 2}],
        "columns": QUOTE_COLS, "sort": {"sortBy": vf, "sortOrder": "desc"}, "range": [0, 300]})
    if not rows:
        return
    prev = state.get("ign_prev", {})
    prev_at = state.get("ign_prev_at")
    gap_min = (now - datetime.fromisoformat(prev_at)).total_seconds() / 60 if prev_at else None
    today = now_et.date().isoformat()
    done = state.setdefault("ign_done", {})
    for d in list(done):
        if d != today:
            del done[d]
    done_today = done.setdefault(today, [])
    recent = state.get("cat_recent", {})
    hits = []
    for r in rows:
        t = r["t"]
        price, chg, vol = live(r, now_et)
        fl, avg = _num(r.get("float_shares_outstanding")), _num(r.get("average_volume_10d_calc"))
        if not price or chg is None or vol is None or t in done_today:
            continue
        if fl and fl > IGN_MAX_FLOAT:
            continue
        if not (IGN_MIN_CHG <= chg <= IGN_MAX_CHG):
            continue
        p0 = prev.get(t)
        if p0 and gap_min is not None and gap_min <= 20:
            step = (price / p0[0] - 1) * 100 if p0[0] else 0
            dvol = vol - p0[1]
            if step < IGN_MIN_STEP or dvol * price < IGN_MIN_DOLLAR_STEP:
                continue
            why = f"{gap_min:.0f}분 새 +{step:.1f}% · 거래량 +{_big(dvol)}"
        elif gap_min is not None and gap_min <= 20:
            if avg and vol < avg * 0.2:          # 새로 등장했지만 거래량이 너무 적으면 제외
                continue
            step, why = chg, f"새로 등장 · 누적 거래량 {_big(vol)}"
        else:
            continue                             # 첫 실행은 기준값만 저장
        cat = recent.get(t)
        prio = (2 if cat else 0) + (1 if fl and fl < 10e6 else 0) + step / 10
        hits.append((prio, t, r, price, chg, vol, fl, avg, why, cat))
    state["ign_prev"] = {r["t"]: list(live(r, now_et)[::2]) for r in rows if live(r, now_et)[0]}
    state["ign_prev_at"] = now.isoformat()
    hits.sort(key=lambda h: -h[0])
    import guard
    sent_n = 0
    for prio, t, r, price, chg, vol, fl, avg, why, cat in hits:
        if sent_n >= IGN_PER_RUN:
            break
        ok, bad, ginfo = guard.pre_check(t, price, chg)
        if not ok:
            continue                                     # 설거지 위험 → 제외
        sent_n += 1
        done_today.append(t)
        rel = f" · 평소의 {vol / avg:.1f}배" if avg else ""
        tag = (f"\n🧨 <b>재료+수급 동시</b>: [{cat['grade']}] {html.escape(cat['label'])} — {html.escape(cat['title'])}"
               if cat else "\n❔ 재료 미확인 — 뉴스 없이 오르면 설거지 주의")
        import signals
        up = "위에 매물 없음" if ginfo.get("upside") is None else f"다음 저항 +{ginfo['upside']}%"
        signals.buy(state, send, t, price, "급등",
                    why=[f"{sess} {chg:+.1f}%", why, f"유통 {_big(fl)}", up,
                         f"재료 [{cat['grade']}] {cat['label']}" if cat else "재료 미확인"],
                    name=r.get("description") or "")
        log_alert(state, "점화", t, price, combo=bool(cat))
