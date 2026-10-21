"""
텔레그램 미국 주식 알림 봇 (GitHub Actions / PC 겸용)
════════════════════════════════════════════
기능
  1) 관심종목 실시간 신호 (5분봉, 프리장 포함)
     - RSI(14) ≤ 30 과매도
     - 전일 대비 거래량 ≥ 200% AND 주가 ≥ +5% 급등
  2) 관심종목 호재/악재 뉴스 알림 (한국어 번역)
  3) 나스닥 소형주 매집 스캔 (하루 2회)
  4) 종배 후보 리포트 (장 마감 10분 전, 조기폐장일 자동 반영)
  5) 미국 증시 휴장일 안내
  6) 급등 후보 (프리장·정규장 5분마다, 설거지 의심 자동 제외)

텔레그램 명령어 (봇에게 보내기, 5분 안에 반영)
  /add TSLA AMD   관심종목 추가
  /del TSLA       관심종목 삭제
  /list           관심종목 보기
  /scan           매집 스캔 바로 실행
  /jongbae        종배 후보 바로 실행
  /help           도움말

환경변수: TELEGRAM_TOKEN(필수), TELEGRAM_CHAT_ID(선택), RUN_ONCE=1(GitHub), TASK=test|scan|jongbae
"""

import html
import json
import logging
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator

# ═════════════════════ 설정 ═════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA"]       # 처음 기본값 (이후 /add /del 로 관리)

# 1) 실시간 신호
RSI_PERIOD = 14
RSI_OVERSOLD = 30
VOLUME_RATIO_MIN = 2.0          # 200%
PRICE_SURGE_MIN = 5.0           # +5%
COOLDOWN = timedelta(hours=1)
PREMARKET = True                # 프리장(4:00 ET~) 감시

# 2) 뉴스
NEWS_MAX_AGE = timedelta(hours=6)
NEWS_INCLUDE_NEUTRAL = False    # 중립 뉴스도 받으려면 True

# 3) 소형주 매집 스캔
SCAN_TIMES_ET = ["10:30", "14:30"]   # 뉴욕 시간 (한국시간 밤 11:30, 새벽 3:30 / 서머타임 기준)
SMALLCAP_MIN = 50e6                  # 시총 5천만 달러
SMALLCAP_MAX = 2e9                   # 시총 20억 달러
SCAN_MIN_PRICE = 1.0
SCAN_MIN_AVG_VOLUME = 300_000
SCAN_MIN_SCORE = 3                   # 5점 만점
SCAN_TOP = 10

# 4) 종배 후보
JONGBAE_BEFORE_CLOSE = 10            # 마감 몇 분 전
JONGBAE_MIN_DROP = -5.0              # 당일 -5% 이상 하락
JONGBAE_MIN_PRICE = 2.0
JONGBAE_MIN_MCAP = 100e6
JONGBAE_MIN_DOLLAR_VOL = 5e6
JONGBAE_TOP = 8

# 5) 급등 후보 (프리장·정규장, 5분마다 나스닥 전체)
SURGE_EXCHANGES = ["NASDAQ"]
SURGE_MIN_CHANGE = 10.0              # +10% 이상 오르기 시작한 종목
SURGE_MAX_CHANGE = 200.0             # +200% 넘게 오른 건 추격위험으로 제외
SURGE_MIN_PRICE = 0.10
SURGE_MAX_PRICE = 20.0
SURGE_MIN_VOLUME = 300_000           # 프리장 누적 거래량(주)
SURGE_MIN_DOLLAR_VOL = 100_000       # 거래대금($)
SURGE_TOP = 8

CHECK_EVERY_SEC = 60                 # PC 모드 주기
RUN_ONCE = os.getenv("RUN_ONCE") == "1"
TASK = os.getenv("TASK", "").strip().lower()
STATE_FILE = Path(__file__).with_name("state.json")
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
# ═══════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stock_bot")
http = requests.Session()
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}


# ───────────────── 상태 ─────────────────
def load_state() -> dict:
    base = {"chat_id": "", "tickers": list(DEFAULT_TICKERS), "last_alert": {},
            "news_seen": [], "done": {}, "update_offset": 0}
    try:
        if STATE_FILE.exists():
            base.update(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except Exception as e:
        log.warning("state.json 읽기 실패: %s", e)
    return base


def save_state() -> None:
    try:
        STATE["news_seen"] = STATE["news_seen"][-500:]
        STATE_FILE.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.error("state.json 저장 실패: %s", e)


STATE = load_state()


def done_today(key: str, day: date) -> bool:
    return key in STATE["done"].get(day.isoformat(), [])


def mark_done(key: str, day: date) -> None:
    d = day.isoformat()
    STATE["done"] = {d: STATE["done"].get(d, []) + [key]}   # 오늘 것만 유지


# ───────────────── 텔레그램 ─────────────────
def tg(method: str, **params) -> dict | None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    for attempt in range(1, 4):
        try:
            r = http.post(url, data=params, timeout=15)
            data = r.json()
            if data.get("ok"):
                return data
            if r.status_code == 429:
                time.sleep(data.get("parameters", {}).get("retry_after", 5))
                continue
            log.error("텔레그램 %s 오류: %s", method, data.get("description"))
            return None
        except Exception as e:
            log.warning("텔레그램 %s 실패 (%d/3): %s", method, attempt, e)
            time.sleep(2 * attempt)
    return None


def send(text: str) -> bool:
    if not TELEGRAM_CHAT_ID:
        return False
    for chunk in split_message(text):
        if not tg("sendMessage", chat_id=TELEGRAM_CHAT_ID, text=chunk,
                  parse_mode="HTML", disable_web_page_preview="true"):
            return False
    return True


def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for block in text.split("\n\n"):
        if len(cur) + len(block) + 2 > limit and cur:
            parts.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + block
    return parts + ([cur] if cur else [])


HELP = ("🤖 <b>사용법</b>\n"
        "/add TSLA AMD — 관심종목 추가\n"
        "/del TSLA — 삭제\n"
        "/list — 목록\n"
        "/scan — 매집 스캔 지금 실행\n"
        "/jongbae — 종배 후보 지금 실행\n"
        "※ 명령은 5분 안에 반영돼요 (한국시간 오후 5시~새벽 6시)")


def process_updates() -> set[str]:
    """봇에게 온 명령 처리. 즉시 실행할 작업 목록 반환"""
    global TELEGRAM_CHAT_ID
    forced: set[str] = set()
    data = tg("getUpdates", offset=STATE.get("update_offset", 0), timeout=0)
    if not data:
        return forced
    for upd in data["result"]:
        STATE["update_offset"] = upd["update_id"] + 1
        msg = upd.get("message") or upd.get("edited_message")
        if not msg or "text" not in msg:
            continue
        chat = str(msg["chat"]["id"])
        if not TELEGRAM_CHAT_ID and not STATE.get("chat_id"):
            STATE["chat_id"] = TELEGRAM_CHAT_ID = chat
            send("🤖 <b>주식 알림 봇 연결 완료!</b>\n\n" + HELP)
        if chat != (TELEGRAM_CHAT_ID or STATE.get("chat_id")):
            continue  # 주인 외 사용자 무시

        parts = msg["text"].strip().split()
        cmd = parts[0].lower().split("@")[0]
        args = [p.upper().replace(",", "") for p in parts[1:]]
        tickers = STATE["tickers"]
        if cmd == "/add" and args:
            added = [a for a in args if re.fullmatch(r"[A-Z.\-]{1,6}", a) and a not in tickers]
            tickers.extend(added)
            send(f"✅ 추가: {', '.join(added) or '없음'}\n📋 관심종목: {', '.join(tickers)}")
        elif cmd == "/del" and args:
            STATE["tickers"] = [t for t in tickers if t not in args]
            send(f"🗑 삭제 완료\n📋 관심종목: {', '.join(STATE['tickers']) or '없음'}")
        elif cmd == "/list":
            send(f"📋 관심종목: {', '.join(tickers) or '없음'}")
        elif cmd == "/scan":
            forced.add("scan")
            send("🧲 매집 스캔을 시작해요. 몇 분 걸려요…")
        elif cmd == "/jongbae":
            forced.add("jongbae")
            send("🎯 종배 후보를 분석해요. 몇 분 걸려요…")
        elif cmd in ("/start", "/help"):
            send(HELP)
    return forced


# ───────────────── 장 시간 / 휴장 ─────────────────
def market_hours(day: date) -> tuple[datetime, datetime] | None:
    """해당 날짜의 정규장 (개장, 마감). 휴장이면 None. 조기폐장 자동 반영"""
    try:
        import pandas_market_calendars as mcal
        sch = mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
        if sch.empty:
            return None
        o = sch.iloc[0]["market_open"].tz_convert(ET).to_pydatetime()
        c = sch.iloc[0]["market_close"].tz_convert(ET).to_pydatetime()
        return o, c
    except ImportError:
        log.warning("pandas_market_calendars 없음 — 휴장일 판단 불가, 평일은 개장으로 간주")
    except Exception as e:
        log.warning("장 일정 조회 실패: %s", e)
    if day.weekday() >= 5:
        return None
    return (datetime.combine(day, dtime(9, 30), ET), datetime.combine(day, dtime(16, 0), ET))


def session(now: datetime, hours) -> str | None:
    if not hours:
        return None
    o, c = hours
    if o <= now <= c:
        return "정규장"
    if PREMARKET and datetime.combine(now.date(), dtime(4, 0), ET) <= now < o:
        return "프리장"
    return None


# ───────────────── 공통 유틸 ─────────────────
def num(x) -> float | None:
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def rsi_last(close: pd.Series, n: int = RSI_PERIOD) -> float:
    try:
        return float(RSIIndicator(close=close, window=n).rsi().iloc[-1])
    except Exception:
        return float("nan")


def wlen(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def table(headers: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> str:
    """폰 화면용 고정폭 표 (한글 폭 보정)"""
    cols = list(zip(headers, *rows))
    widths = [max(wlen(str(v)) for v in col) for col in cols]

    def fmt(r):
        out = []
        for i, v in enumerate(r):
            v = str(v)
            pad = " " * (widths[i] - wlen(v))
            out.append(pad + v if i in right else v + pad)
        return " ".join(out).rstrip()

    lines = [fmt(headers), "─" * (sum(widths) + len(widths) - 1)] + [fmt(r) for r in rows]
    return "<pre>" + html.escape("\n".join(lines)) + "</pre>"


def price_str(p: float) -> str:
    return f"{p:,.2f}" if p < 1000 else f"{p:,.0f}"


def translate_ko(text: str) -> str:
    try:
        r = http.get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": text},
                     timeout=8)
        return "".join(p[0] for p in r.json()[0] if p and p[0]) or text
    except Exception:
        return text


def nasdaq_universe() -> list[dict]:
    """나스닥 전 종목 (시총/가격/등락률) — 요청 1번으로 받음"""
    try:
        r = http.get("https://api.nasdaq.com/api/screener/stocks",
                     params={"tableonly": "true", "limit": "10000", "exchange": "nasdaq",
                             "download": "true"}, headers=UA, timeout=30)
        rows = r.json()["data"]["rows"]
    except Exception as e:
        log.error("나스닥 종목 목록 조회 실패: %s", e)
        return []
    out = []
    for row in rows:
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym or "^" in sym or (len(sym) == 5 and sym[-1] in "WRU"):
            continue  # 워런트/권리/유닛 제외
        out.append({"symbol": sym.replace("/", "-"), "name": row.get("name", ""),
                    "price": num(row.get("lastsale")), "pct": num(row.get("pctchange")),
                    "volume": num(row.get("volume")), "mcap": num(row.get("marketCap"))})
    log.info("나스닥 종목 %d개", len(out))
    return out


def download_daily(tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 150):
        chunk = tickers[i:i + 150]
        try:
            df = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                             auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            log.warning("일봉 다운로드 실패 (%d~): %s", i, e)
            continue
        if df is None or df.empty:
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


# ───────────────── 뉴스 ─────────────────
POS_WORDS = ["beat", "beats", "tops", "surpass", "raises guidance", "raised guidance", "upgrade",
             "record", "approval", "approved", "approves", "partnership", "contract", "awarded",
             "buyback", "repurchase", "soar", "surge", "jump", "rally", "breakthrough",
             "outperform", "raises price target", "price target raised", "wins", "to be acquired",
             "acquisition by", "strong demand", "expands", "launches", "positive results"]
NEG_WORDS = ["miss", "misses", "cuts guidance", "lowers guidance", "downgrade", "lawsuit", "sued",
             "probe", "investigation", "subpoena", "recall", "plunge", "tumble", "sink", "slump",
             "falls", "drops", "decline", "offering", "dilution", "bankruptcy", "chapter 11",
             "delist", "layoff", "halt", "fraud", "warning", "weak", "disappoint", "resigns",
             "short seller", "rejects", "complete response letter", "fails", "failed",
             "lowers price target", "price target cut", "going concern", "default"]


def _count(text: str, words: list[str]) -> int:
    return sum(1 for w in words if re.search(r"\b" + re.escape(w), text))


def sentiment(title: str) -> str:
    t = title.lower()
    p, n = _count(t, POS_WORDS), _count(t, NEG_WORDS)
    return "호재" if p > n else "악재" if n > p else "중립"


def fetch_news(ticker: str) -> list[dict]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        log.warning("%s 뉴스 조회 실패: %s", ticker, e)
        return []
    out = []
    for it in items:
        try:
            c = it.get("content") or it
            nid = it.get("id") or c.get("id") or it.get("uuid")
            title = c.get("title") or ""
            pub = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
            link = ((c.get("canonicalUrl") or {}).get("url")
                    or (c.get("clickThroughUrl") or {}).get("url") or c.get("link") or "")
            if c.get("pubDate"):
                ts = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            elif it.get("providerPublishTime"):
                ts = datetime.fromtimestamp(it["providerPublishTime"], timezone.utc)
            else:
                continue
            summary = c.get("summary") or ""
            if nid and title:
                out.append({"id": nid, "title": title, "pub": pub, "link": link,
                            "ts": ts, "summary": summary})
        except Exception:
            continue
    return out


def check_news() -> None:
    now = datetime.now(timezone.utc)
    seen = set(STATE["news_seen"])
    items = []
    for t in STATE["tickers"]:
        for n in fetch_news(t):
            if n["id"] in seen or now - n["ts"] > NEWS_MAX_AGE:
                continue
            STATE["news_seen"].append(n["id"])
            seen.add(n["id"])
            s = sentiment(n["title"])
            if s == "중립" and not NEWS_INCLUDE_NEUTRAL:
                continue
            items.append((t, s, n))
    if not items:
        return
    icon = {"호재": "🟢", "악재": "🔴", "중립": "⚪"}
    blocks = []
    for t, s, n in sorted(items, key=lambda x: x[2]["ts"], reverse=True)[:10]:
        ko = translate_ko(n["title"])
        when = n["ts"].astimezone(KST).strftime("%m/%d %H:%M")
        link = f'<a href="{html.escape(n["link"])}">원문</a> · ' if n["link"] else ""
        blocks.append(f"{icon[s]} <b>{s} | {t}</b>\n{html.escape(ko)}\n{link}{html.escape(n['pub'])} · {when}")
    send("📰 <b>관심종목 뉴스</b>\n━━━━━━━━━━━━━━━\n" + "\n\n".join(blocks))


# ───────────────── 1) 관심종목 실시간 신호 ─────────────────
def fetch_5m(ticker: str) -> tuple[pd.DataFrame, float] | None:
    try:
        t = yf.Ticker(ticker)
        bars = t.history(period="5d", interval="5m", prepost=PREMARKET)
        daily = t.history(period="10d", interval="1d")
    except Exception as e:
        log.warning("%s 데이터 조회 실패: %s", ticker, e)
        return None
    if bars is None or bars.empty or daily is None or len(daily) < 2:
        return None
    bars = bars.dropna(subset=["Close", "Volume"])
    bars.index = bars.index.tz_convert(ET)
    daily.index = daily.index.tz_convert(ET)
    prev = daily[daily.index.date < bars.index[-1].date()]
    if prev.empty:
        return None
    return bars, float(prev["Close"].iloc[-1])


def analyze(bars: pd.DataFrame, prev_close: float) -> dict | None:
    if len(bars) < RSI_PERIOD + 1:
        return None
    rsi = rsi_last(bars["Close"])
    price = float(bars["Close"].iloc[-1])
    change = (price - prev_close) / prev_close * 100
    dates = sorted(set(bars.index.date))
    vol_ratio = None
    if len(dates) >= 2:
        tb = bars[bars.index.date == dates[-1]]
        pb = bars[bars.index.date == dates[-2]]
        same = pb[pb.index.time <= tb.index[-1].time()]["Volume"].sum()
        if same > 0:
            vol_ratio = tb["Volume"].sum() / same
    signals = []
    if pd.notna(rsi) and rsi <= RSI_OVERSOLD:
        signals.append(f"🔻 <b>과매도</b> (RSI {rsi:.1f})")
    if vol_ratio and vol_ratio >= VOLUME_RATIO_MIN and change >= PRICE_SURGE_MIN:
        signals.append(f"🔥 <b>거래량·주가 급등</b> (거래량 {vol_ratio * 100:.0f}%)")
    return {"price": price, "change": change, "rsi": rsi, "vol_ratio": vol_ratio, "signals": signals}


def check_realtime(sess: str) -> None:
    for t in STATE["tickers"]:
        try:
            last = STATE["last_alert"].get(t)
            if last and datetime.now(timezone.utc) - datetime.fromisoformat(last) < COOLDOWN:
                continue
            data = fetch_5m(t)
            r = analyze(*data) if data else None
            if not r or not r["signals"]:
                continue
            arrow = "📈" if r["change"] >= 0 else "📉"
            vol = f"{r['vol_ratio'] * 100:.0f}%" if r["vol_ratio"] else "N/A"
            msg = (f"🚨 <b>{t} 신호 포착</b> ({'🌅' if sess == '프리장' else '🔔'} {sess})\n"
                   f"💰 ${price_str(r['price'])}  {arrow} {r['change']:+.2f}%\n"
                   f"📊 RSI {r['rsi']:.1f} · 거래량 {vol}\n" + "\n".join(r["signals"]))
            if send(msg):
                STATE["last_alert"][t] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            log.exception("%s 실시간 체크 오류: %s", t, e)


# ───────────────── 3) 소형주 매집 스캔 ─────────────────
def accumulation_score(df: pd.DataFrame) -> dict | None:
    """O 매집(OBV↑·횡보) / U 상승일 거래량 우위 / S 변동성 수축 / B 돌파 임박 / V 거래량 증가"""
    if len(df) < 60:
        return None
    c, h, v = df["Close"], df["High"], df["Volume"]
    price = float(c.iloc[-1])
    avg20 = float(v.iloc[-21:-1].mean())
    if price < SCAN_MIN_PRICE or avg20 < SCAN_MIN_AVG_VOLUME:
        return None
    chg20 = (price / float(c.iloc[-21]) - 1) * 100
    chg1 = (price / float(c.iloc[-2]) - 1) * 100
    if chg20 > 30 or chg1 > 15:      # 이미 급등한 종목 제외
        return None

    flags = ""
    direction = c.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    obv = (direction * v).cumsum()
    if (obv.iloc[-1] - obv.iloc[-21]) / max(v.iloc[-20:].sum(), 1) > 0.15 and -8 <= chg20 <= 10:
        flags += "O"
    up_v = v.iloc[-20:][direction.iloc[-20:] > 0].sum()
    dn_v = v.iloc[-20:][direction.iloc[-20:] < 0].sum()
    if dn_v > 0 and up_v / dn_v >= 1.5:
        flags += "U"
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    bw = ((4 * sd) / ma).dropna().iloc[-120:]
    if len(bw) > 20 and bw.iloc[-1] <= bw.quantile(0.2):
        flags += "S"
    if price >= 0.95 * float(h.iloc[-61:-1].max()):
        flags += "B"
    if avg20 > 0 and float(v.iloc[-1]) / avg20 >= 1.5:
        flags += "V"

    if len(flags) < SCAN_MIN_SCORE or not ({"O", "U"} & set(flags)):
        return None
    return {"price": price, "chg1": chg1, "score": len(flags), "flags": flags,
            "vratio": float(v.iloc[-1]) / avg20 if avg20 else 0}


def run_scan(label: str) -> None:
    uni = nasdaq_universe()
    small = [u["symbol"] for u in uni
             if u["mcap"] and SMALLCAP_MIN <= u["mcap"] <= SMALLCAP_MAX
             and (u["price"] or 0) >= SCAN_MIN_PRICE and (u["volume"] or 0) >= 50_000]
    if not small:
        send("⚠️ 나스닥 종목 목록을 받지 못해 매집 스캔을 건너뛰었어요.")
        return
    log.info("매집 스캔 대상 %d개", len(small))
    data = download_daily(small)
    hits = []
    for t, df in data.items():
        try:
            r = accumulation_score(df)
            if r:
                hits.append((t, r))
        except Exception:
            continue
    hits.sort(key=lambda x: (x[1]["score"], x[1]["vratio"]), reverse=True)
    hits = hits[:SCAN_TOP]

    head = f"🧲 <b>소형주 매집 포착</b> ({label})\n<i>나스닥 시총 5천만~20억$ · {len(data)}종목 분석</i>\n"
    if not hits:
        send(head + "\n오늘은 조건에 맞는 종목이 없어요.")
        return
    rows = [[str(i), t, price_str(r["price"]), f"{r['chg1']:+.1f}%", f"{r['score']}/5", r["flags"]]
            for i, (t, r) in enumerate(hits, 1)]
    send(head + table(["#", "티커", "주가", "등락", "점수", "신호"], rows, right={2, 3})
         + "\nO 매집 · U 상승거래 우위 · S 수축\nB 돌파임박 · V 거래량↑\n⚠️ 투자 판단은 본인 책임")


# ───────────────── 4) 종배 후보 ─────────────────
CAUSES = [  # (키워드, 원인, 위험 기본점수)
    (["offering", "dilution", "private placement", "registered direct", "warrant", "at-the-market",
      "priced"], "증자·희석", 3),
    (["bankruptcy", "chapter 11", "delist", "going concern", "default"], "상폐·파산", 4),
    (["trial", "fda", "complete response", "phase"], "임상·FDA", 3),
    (["lawsuit", "probe", "investigation", "subpoena", "sec ", "fraud", "short seller"], "소송·조사", 3),
    (["guidance", "outlook", "forecast"], "가이던스↓", 2),
    (["miss", "earnings", "revenue", "loss", "quarter", "results"], "실적부진", 2),
    (["downgrade", "price target", "cut to"], "등급하향", 1),
    (["ceo", "resign", "layoff", "restructur"], "경영이슈", 2),
    (["tariff", "sector", "china", "rates", "market", "selloff"], "업황·매크로", 1),
]


def drop_cause(ticker: str) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    news = [n for n in fetch_news(ticker) if now - n["ts"] < timedelta(hours=36)]
    text = " ".join((n["title"] + " " + n["summary"]).lower() for n in news)
    for words, cause, risk in CAUSES:
        if any(w in text for w in words):
            return cause, risk
    return ("원인불명", 2) if not news else ("기타뉴스", 2)


def oversold_stage(rsi: float) -> int:
    if pd.isna(rsi):
        return 0
    return 3 if rsi <= 20 else 2 if rsi <= 30 else 1 if rsi <= 40 else 0


def run_jongbae(label: str) -> None:
    uni = nasdaq_universe()
    cands = [u for u in uni if u["pct"] is not None and u["pct"] <= JONGBAE_MIN_DROP
             and (u["price"] or 0) >= JONGBAE_MIN_PRICE and (u["mcap"] or 0) >= JONGBAE_MIN_MCAP
             and (u["price"] or 0) * (u["volume"] or 0) >= JONGBAE_MIN_DOLLAR_VOL]
    mcap = {u["symbol"]: u["mcap"] for u in uni}
    tickers = [u["symbol"] for u in sorted(cands, key=lambda u: u["pct"])[:150]]
    tickers += [t for t in STATE["tickers"] if t not in tickers]   # 관심종목도 포함
    note = "" if uni else "\n<i>나스닥 목록 조회 실패 — 관심종목만 분석</i>"

    data = download_daily(tickers, period="3mo")
    res = []
    for t, df in data.items():
        try:
            price, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            drop = (price / prev - 1) * 100
            if drop > JONGBAE_MIN_DROP:
                continue
            rsi = rsi_last(df["Close"])
            stage = oversold_stage(rsi)
            if stage:
                res.append({"t": t, "price": price, "drop": drop, "rsi": rsi, "stage": stage})
        except Exception:
            continue
    res.sort(key=lambda r: (-r["stage"], r["rsi"]))
    res = res[:JONGBAE_TOP]

    head = f"🎯 <b>종배 후보</b> ({label}){note}\n"
    if not res:
        send(head + "\n오늘은 조건에 맞는 종배 후보가 없어요.")
        return

    t1 = [[str(i), r["t"], price_str(r["price"]), f"{r['drop']:.1f}%", f"{r['stage']}단계"]
          for i, r in enumerate(res, 1)]
    t2 = []
    for r in res:
        cause, risk = drop_cause(r["t"])
        risk += (r["drop"] <= -20) + ((mcap.get(r["t"]) or 1e12) < 300e6) + (r["price"] < 5)
        t2.append([r["t"], cause, "높음" if risk >= 4 else "중간" if risk == 3 else "낮음"])

    send(head
         + "\n📉 <b>과매도 순위</b>\n" + table(["순위", "티커", "주가", "하락률", "단계"], t1, right={2, 3})
         + "단계: 3 RSI≤20 · 2 ≤30 · 1 ≤40\n"
         + "\n⚠️ <b>급락 원인 · 위험도</b>\n" + table(["티커", "급락원인", "위험도"], t2)
         + "투자 판단은 본인 책임")


# ───────────────── 5) 급등 후보 + 설거지 필터 ─────────────────
def tv_scan(sess: str) -> list[dict]:
    """TradingView 스캐너로 나스닥 급등 종목을 요청 1번에 조회"""
    pre = sess == "프리장"
    chg, vol, px = (("premarket_change", "premarket_volume", "premarket_close") if pre
                    else ("change", "volume", "close"))
    cols = [px, chg, vol, "average_volume_10d_calc", "market_cap_basic", "float_shares_outstanding"]
    body = {
        "markets": ["america"],
        "filter": [
            {"left": "exchange", "operation": "in_range", "right": SURGE_EXCHANGES},
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": chg, "operation": "greater", "right": SURGE_MIN_CHANGE},
            {"left": px, "operation": "in_range", "right": [SURGE_MIN_PRICE, SURGE_MAX_PRICE]},
            {"left": vol, "operation": "greater", "right": SURGE_MIN_VOLUME},
        ],
        "columns": cols,
        "sort": {"sortBy": chg, "sortOrder": "desc"},
        "range": [0, 60],
    }
    for attempt in range(2):
        try:
            r = http.post("https://scanner.tradingview.com/america/scan", json=body,
                          headers=UA, timeout=20)
            data = r.json()
            if "data" not in data:
                raise ValueError(str(data)[:200])
            out = []
            for row in data["data"]:
                d = dict(zip(cols, row["d"]))
                out.append({"t": row["s"].split(":")[-1].replace(".", "-"),
                            "price": num(d.get(px)), "chg": num(d.get(chg)), "vol": num(d.get(vol)),
                            "avgvol": num(d.get("average_volume_10d_calc")),
                            "float": num(d.get("float_shares_outstanding"))})
            return out
        except Exception as e:
            log.warning("급등 스캐너 조회 실패 (%d/2): %s", attempt + 1, e)
            if "float_shares_outstanding" in cols:   # 필드 오류 대비 최소 컬럼으로 재시도
                cols.remove("float_shares_outstanding")
            time.sleep(2)
    return []


DUMP_NEWS = ["offering", "registered direct", "private placement", "warrant", "at-the-market",
             "reverse split", "reverse stock split", "dilution", "shelf", "delist", "going concern"]


def dump_check(t: str, chg: float) -> tuple[bool, list[str], list[str]]:
    """설거지 위험 판정 → (통과여부, 좋은 신호, 위험 신호)"""
    good, bad = [], []
    hard = False
    try:
        bars = yf.Ticker(t).history(period="1d", interval="1m", prepost=True)
        bars = bars.dropna(subset=["Close"])
    except Exception as e:
        log.warning("%s 1분봉 실패: %s", t, e)
        bars = pd.DataFrame()

    if len(bars) >= 10:
        c, o, h, v = bars["Close"], bars["Open"], bars["High"], bars["Volume"]
        last, hi = float(c.iloc[-1]), float(h.max())
        tp = (bars["High"] + bars["Low"] + c) / 3
        vwap = float((tp * v).sum() / max(v.sum(), 1))
        red = float(v.iloc[-15:][c.iloc[-15:] < o.iloc[-15:]].sum())
        green = float(v.iloc[-15:][c.iloc[-15:] > o.iloc[-15:]].sum())
        dd = (hi - last) / hi * 100 if hi else 0

        if dd > 25:
            bad.append(f"고점대비-{dd:.0f}%"); hard = True
        elif last >= hi * 0.92:
            good.append("고점근접")
        if last < vwap:
            bad.append("VWAP이탈")
            if red > green * 1.5:
                hard = True
        elif last > vwap * 1.03:
            good.append("VWAP위")
        if red > green * 1.5 and red > 0:
            bad.append("매도우위")
        elif green > red * 1.5:
            good.append("매수우위")
        # 최근 15분 저점이 높아지는지 (계단식 상승)
        lows = bars["Low"].iloc[-15:]
        if len(lows) == 15 and lows.iloc[10:].min() > lows.iloc[:5].min():
            good.append("저점상승")

    if chg > SURGE_MAX_CHANGE:
        bad.append("과열"); hard = True

    # 과거 설거지 이력: 장중 +50% 솟았다가 +10% 아래로 마감한 날
    try:
        d = yf.Ticker(t).history(period="3mo", interval="1d")
        if len(d) > 5:
            d = d.iloc[:-1]
            pumps = ((d["High"] / d["Open"] - 1 >= 0.5) & (d["Close"] / d["Open"] - 1 <= 0.1)).sum()
            if pumps >= 2:
                bad.append(f"설거지이력{pumps}회"); hard = True
    except Exception:
        pass

    # 뉴스: 증자/병합은 제외, 호재는 가점
    now = datetime.now(timezone.utc)
    news = [n for n in fetch_news(t) if now - n["ts"] < timedelta(days=3)]
    text = " ".join((n["title"] + " " + n["summary"]).lower() for n in news)
    if any(w in text for w in DUMP_NEWS):
        bad.append("증자·병합"); hard = True
    elif any(sentiment(n["title"]) == "호재" for n in news):
        good.append("호재뉴스")

    return (not hard and len(good) >= 2), good, bad


def run_surge(sess: str) -> None:
    today = datetime.now(ET).date().isoformat()
    alerted = STATE.setdefault("surge_alerted", {})
    alerted_today = alerted.setdefault(today, [])
    for d in list(alerted):
        if d != today:
            del alerted[d]

    rows = tv_scan(sess)
    cands = []
    for r in rows:
        if r["t"] in alerted_today or not r["price"] or not r["vol"]:
            continue
        if r["price"] * r["vol"] < SURGE_MIN_DOLLAR_VOL:
            continue
        rel = r["vol"] / r["avgvol"] if r["avgvol"] else 0
        if sess == "프리장" and r["avgvol"] and rel < 0.2:   # 평소 거래량의 20%도 안 되면 제외
            continue
        if sess == "정규장" and rel < 2:
            continue
        r["rel"] = rel
        cands.append(r)
    cands.sort(key=lambda r: r["rel"], reverse=True)

    passed, dumped = [], []
    for r in cands[:15]:
        ok, good, bad = dump_check(r["t"], r["chg"])
        if ok:
            passed.append((r, good, bad))
        else:
            dumped.append(r["t"])
        alerted_today.append(r["t"])          # 한 번 본 종목은 오늘 다시 안 봄
        if len(passed) >= SURGE_TOP:
            break
    if not passed:
        return

    def vol_s(x):
        return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"

    tbl = []
    for r, good, bad in passed:
        risk = "높음" if len(bad) >= 2 else "중간" if bad or r["price"] < 1 else "낮음"
        p = f"{r['price']:.3f}" if r["price"] < 1 else f"{r['price']:.2f}"
        tbl.append([r["t"], p, f"+{r['chg']:.0f}%", vol_s(r["vol"]), risk])
    why = "\n".join(f"• <b>{r['t']}</b> {'·'.join(g)}"
                    + (f" ⚠️{'·'.join(b)}" if b else "")
                    + (f" (유통 {r['float'] / 1e6:.1f}M주)" if r.get("float") else "")
                    for r, g, b in passed)
    icon = "🌅" if sess == "프리장" else "🔔"
    kst = datetime.now(KST).strftime("%H:%M")
    msg = (f"🚀 <b>급등 후보</b> ({icon} {sess} {kst})\n"
           + table(["티커", "주가", "등락", "거래량", "위험"], tbl, right={1, 2, 3})
           + why)
    if dumped:
        msg += f"\n\n🚫 설거지 의심 제외: {', '.join(dumped)}"
    send(msg + "\n⚠️ 저가주는 변동성 극심 · 손절 필수")


# ───────────────── 실행 ─────────────────
def run_once() -> None:
    global TELEGRAM_CHAT_ID
    TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID or STATE.get("chat_id", "")
    forced = process_updates()
    if TASK in ("scan", "jongbae"):
        forced.add(TASK)
    if not TELEGRAM_CHAT_ID:
        log.warning("채팅 ID 없음 — 텔레그램에서 봇에게 /start 를 보내주세요.")
        return

    now = datetime.now(ET)
    today = now.date()
    hours = market_hours(today)
    sess = session(now, hours)

    if TASK == "test":
        status = (f"🟢 {sess} 감시 중" if sess else "🏖 오늘은 미국 증시 휴장일이에요" if not hours
                  else "⚪ 장 시간 아님 — 장이 열리면 자동 감시")
        send(f"✅ <b>연결 테스트 성공</b>\n{status}\n📋 관심종목: {', '.join(STATE['tickers'])}")

    if not hours:
        if today.weekday() < 5 and not done_today("holiday", today):
            send(f"🏖 오늘({today:%m/%d})은 <b>미국 증시 휴장일</b>이에요.\n알림은 다음 거래일에 다시 시작해요.")
            mark_done("holiday", today)
        if forced:
            send("🏖 오늘은 휴장일이라 스캔을 건너뛰어요.")
        return

    o, c = hours
    try:
        check_news()
    except Exception as e:
        log.exception("뉴스 오류: %s", e)

    if sess:
        check_realtime(sess)
        try:
            run_surge(sess)
        except Exception as e:
            log.exception("급등 후보 오류: %s", e)

    # 매집 스캔 (정해진 시각 + 25분 이내, 하루 1번씩)
    for hhmm in SCAN_TIMES_ET:
        target = datetime.combine(today, dtime.fromisoformat(hhmm), ET)
        if ("scan" in forced or target <= now < target + timedelta(minutes=25)) \
                and (("scan" in forced) or not done_today(f"scan{hhmm}", today)):
            try:
                run_scan(now.astimezone(KST).strftime("%m/%d %H:%M"))
            except Exception as e:
                log.exception("매집 스캔 오류: %s", e)
            if "scan" not in forced:
                mark_done(f"scan{hhmm}", today)
            forced.discard("scan")
            break

    # 종배 (마감 N분 전 ~ 마감)
    jb_at = c - timedelta(minutes=JONGBAE_BEFORE_CLOSE)
    if "jongbae" in forced or (jb_at - timedelta(minutes=3) <= now < c and not done_today("jongbae", today)):
        half = " · 조기폐장일" if c.time() < dtime(16, 0) else ""
        try:
            run_jongbae(f"마감 {max(int((c - now).total_seconds() // 60), 0)}분 전{half}")
        except Exception as e:
            log.exception("종배 오류: %s", e)
        if "jongbae" not in forced:
            mark_done("jongbae", today)


def main() -> None:
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN 이 없습니다. (GitHub: Settings → Secrets)")
        return
    if RUN_ONCE:
        try:
            run_once()
        finally:
            save_state()
        return
    log.info("PC 모드 시작")
    while True:
        try:
            run_once()
        except Exception as e:
            log.exception("루프 오류 (계속 실행): %s", e)
        finally:
            save_state()
        time.sleep(CHECK_EVERY_SEC - datetime.now().second % CHECK_EVERY_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("종료")
