"""
텔레그램 주식 알림 봇 (GitHub Actions / PC 겸용)
────────────────────────────────────────────
알림 조건 (5분봉 기준, 둘 중 하나라도 충족 시)
  ① 과매도 : RSI(14) ≤ 30
  ② 급등   : 전일 대비 거래량 ≥ 200%  AND  전일 종가 대비 주가 ≥ +5%
     (거래량은 "오늘 누적 vs 어제 같은 시각까지 누적"으로 비교)

실행 방식
  - GitHub Actions : RUN_ONCE=1 → 한 번 검사하고 종료 (5분마다 자동 실행)
  - PC             : 그냥 실행 → 장중 1분마다 무한 반복
  - 쿨타임/채팅ID는 state.json 에 저장되어 실행 사이에 유지됨

토큰은 코드에 넣지 않고 환경변수(GitHub Secrets)로 받습니다.
  TELEGRAM_TOKEN   : 필수
  TELEGRAM_CHAT_ID : 선택 (없으면 봇에게 보낸 메시지에서 자동 탐지)
"""

import html
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
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

MARKET = "US"                               # "US" 또는 "KR"
TICKERS = ["AAPL", "TSLA", "NVDA"]          # KR 예: ["005930", "000660", "035720"]

RSI_PERIOD = 14
RSI_OVERSOLD = 30
VOLUME_RATIO_MIN = 2.0                      # 200%
PRICE_SURGE_MIN = 5.0                       # +5%
COOLDOWN = timedelta(hours=1)               # 종목당 쿨타임
CHECK_EVERY_SEC = 60                        # PC 실행 시 주기

RUN_ONCE = os.getenv("RUN_ONCE") == "1"     # GitHub Actions 모드
TEST_MESSAGE = os.getenv("TEST_MESSAGE") == "1"  # 수동 실행 시 연결 테스트
STATE_FILE = Path(__file__).with_name("state.json")

MARKETS = {
    "US": {"tz": ZoneInfo("America/New_York"), "open": dtime(9, 30), "close": dtime(16, 0)},
    "KR": {"tz": ZoneInfo("Asia/Seoul"), "open": dtime(9, 0), "close": dtime(15, 30)},
}
# ═══════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stock_bot")
http = requests.Session()
name_cache: dict[str, str] = {}


# ───────────────── 상태 저장 (쿨타임·채팅ID) ─────────────────
def load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("state.json 읽기 실패, 새로 시작: %s", e)
    return {"chat_id": "", "last_alert": {}}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.error("state.json 저장 실패: %s", e)


STATE = load_state()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def in_cooldown(ticker: str) -> bool:
    t = STATE["last_alert"].get(ticker)
    if not t:
        return False
    try:
        return now_utc() - datetime.fromisoformat(t) < COOLDOWN
    except ValueError:
        return False


def mark_alert(ticker: str) -> None:
    STATE["last_alert"][ticker] = now_utc().isoformat()
    save_state(STATE)


# ───────────────── 텔레그램 ─────────────────
def send_telegram(text: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(1, retries + 1):
        try:
            r = http.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                log.warning("텔레그램 속도 제한, %s초 대기", wait)
                time.sleep(wait)
                continue
            log.error("텔레그램 응답 오류 %s: %s", r.status_code, r.text[:200])
            if 400 <= r.status_code < 500:
                return False
        except requests.exceptions.RequestException as e:
            log.warning("텔레그램 전송 실패 (%d/%d): %s", attempt, retries, e)
        time.sleep(2 * attempt)
    return False


def find_chat_id(wait_forever: bool) -> str:
    """봇에게 보낸 최근 메시지에서 채팅 ID 탐지"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    while True:
        try:
            data = http.get(url, timeout=10).json()
            if not data.get("ok"):
                log.error("토큰 오류: %s", data.get("description"))
                return ""
            for upd in reversed(data["result"]):
                msg = upd.get("message") or upd.get("edited_message")
                if msg:
                    return str(msg["chat"]["id"])
            log.warning("봇에게 보낸 메시지가 없습니다. 텔레그램에서 봇에게 /start 를 보내주세요.")
        except Exception as e:
            log.warning("채팅 ID 조회 실패: %s", e)
        if not wait_forever:
            return ""
        time.sleep(10)


def resolve_chat_id() -> bool:
    """채팅 ID 확보. 처음 찾았으면 가동 메시지 전송"""
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID:
        return True
    if STATE.get("chat_id"):
        TELEGRAM_CHAT_ID = STATE["chat_id"]
        return True
    TELEGRAM_CHAT_ID = find_chat_id(wait_forever=not RUN_ONCE)
    if not TELEGRAM_CHAT_ID:
        return False
    STATE["chat_id"] = TELEGRAM_CHAT_ID
    save_state(STATE)
    send_telegram(f"🤖 <b>주식 알림 봇 연결 완료!</b>\n감시 종목: {', '.join(TICKERS)} ({MARKET})\n"
                  f"장중에 신호가 잡히면 알려드릴게요.")
    return True


# ───────────────── 데이터 ─────────────────
def to_symbol(ticker: str) -> str:
    if MARKET == "KR" and "." not in ticker:
        for suffix in (".KS", ".KQ"):
            try:
                if not yf.Ticker(ticker + suffix).history(period="5d").empty:
                    return ticker + suffix
            except Exception:
                pass
        return ticker + ".KS"
    return ticker


def get_name(ticker: str, symbol: str) -> str:
    if ticker in name_cache:
        return name_cache[ticker]
    name = ticker
    try:
        if MARKET == "KR":
            import FinanceDataReader as fdr
            listing = fdr.StockListing("KRX")
            row = listing[listing["Code"] == ticker]
            if not row.empty:
                name = row.iloc[0]["Name"]
        else:
            info = yf.Ticker(symbol).info
            name = info.get("shortName") or info.get("longName") or ticker
    except Exception as e:
        log.debug("종목명 조회 실패 %s: %s", ticker, e)
    name_cache[ticker] = name
    return name


def fetch_data(symbol: str) -> tuple[pd.DataFrame, float] | None:
    try:
        t = yf.Ticker(symbol)
        bars = t.history(period="5d", interval="5m", prepost=False)
        daily = t.history(period="10d", interval="1d")
    except Exception as e:
        log.warning("%s 데이터 조회 실패: %s", symbol, e)
        return None
    if bars is None or bars.empty or daily is None or len(daily) < 2:
        log.warning("%s 데이터 부족", symbol)
        return None

    bars = bars.dropna(subset=["Close", "Volume"])
    tz = MARKETS[MARKET]["tz"]
    bars.index = bars.index.tz_convert(tz)
    daily.index = daily.index.tz_convert(tz)

    today = bars.index[-1].date()
    prev_daily = daily[daily.index.date < today]
    if prev_daily.empty:
        return None
    return bars, float(prev_daily["Close"].iloc[-1])


# ───────────────── 신호 판정 ─────────────────
def analyze(bars: pd.DataFrame, prev_close: float) -> dict | None:
    if len(bars) < RSI_PERIOD + 1:
        return None
    rsi = RSIIndicator(close=bars["Close"], window=RSI_PERIOD).rsi().iloc[-1]
    price = float(bars["Close"].iloc[-1])
    change_pct = (price - prev_close) / prev_close * 100

    dates = sorted(set(bars.index.date))
    vol_ratio = None
    if len(dates) >= 2:
        today_bars = bars[bars.index.date == dates[-1]]
        prev_bars = bars[bars.index.date == dates[-2]]
        now_t = today_bars.index[-1].time()
        prev_same = prev_bars[prev_bars.index.time <= now_t]["Volume"].sum()
        if prev_same > 0:
            vol_ratio = today_bars["Volume"].sum() / prev_same

    signals = []
    if pd.notna(rsi) and rsi <= RSI_OVERSOLD:
        signals.append(f"🔻 <b>과매도</b> (RSI {rsi:.1f} ≤ {RSI_OVERSOLD})")
    if vol_ratio is not None and vol_ratio >= VOLUME_RATIO_MIN and change_pct >= PRICE_SURGE_MIN:
        signals.append(f"🔥 <b>거래량 급등 + 주가 급등</b> (거래량 {vol_ratio * 100:.0f}%, {change_pct:+.2f}%)")

    return {"price": price, "change_pct": change_pct, "rsi": rsi,
            "vol_ratio": vol_ratio, "signals": signals}


def format_message(name: str, ticker: str, r: dict) -> str:
    price = f"{r['price']:,.0f}원" if MARKET == "KR" else f"${r['price']:,.2f}"
    arrow = "📈" if r["change_pct"] >= 0 else "📉"
    vol = f"{r['vol_ratio'] * 100:.0f}%" if r["vol_ratio"] is not None else "N/A"
    kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m/%d %H:%M")
    return (
        f"🚨 <b>주식 신호 포착</b> 🚨\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏷 <b>{html.escape(name)}</b> ({html.escape(ticker)})\n"
        f"💰 현재가: <b>{price}</b>\n"
        f"{arrow} 등락률: <b>{r['change_pct']:+.2f}%</b>\n"
        f"📊 RSI(14): {r['rsi']:.1f}  |  거래량(전일비): {vol}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"⚡ <b>포착된 신호</b>\n" + "\n".join(f"  • {s}" for s in r["signals"]) +
        f"\n━━━━━━━━━━━━━━━\n🕒 {kst} (한국시간)"
    )


# ───────────────── 실행 ─────────────────
def is_market_open() -> bool:
    m = MARKETS[MARKET]
    now = datetime.now(m["tz"])
    return now.weekday() < 5 and m["open"] <= now.time() <= m["close"]


def check_all(symbols: dict[str, str]) -> None:
    for ticker, symbol in symbols.items():
        try:
            if in_cooldown(ticker):
                log.info("%s 쿨타임 중 — 건너뜀", ticker)
                continue
            data = fetch_data(symbol)
            if not data:
                continue
            r = analyze(*data)
            if not r:
                continue
            log.info("%s 가격 %.2f (%+.2f%%) RSI %.1f 거래량비 %s", ticker, r["price"],
                     r["change_pct"], r["rsi"], f"{r['vol_ratio']:.2f}" if r["vol_ratio"] else "N/A")
            if r["signals"] and send_telegram(format_message(get_name(ticker, symbol), ticker, r)):
                mark_alert(ticker)
                log.info("🔔 %s 알림 전송", ticker)
        except Exception as e:
            log.exception("%s 처리 중 오류: %s", ticker, e)


def main() -> None:
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN 이 없습니다. (GitHub: Settings → Secrets 에 등록)")
        return
    if not resolve_chat_id():
        return
    if TEST_MESSAGE:
        status = "🟢 장 열림 — 감시 중" if is_market_open() else "⚪ 장 마감 — 장이 열리면 자동 감시"
        send_telegram(f"✅ <b>연결 테스트 성공</b>\n{status}\n감시 종목: {', '.join(TICKERS)}")

    symbols = {}
    for t in TICKERS:
        try:
            symbols[t] = to_symbol(t)
        except Exception as e:
            log.error("%s 심볼 변환 실패: %s", t, e)

    if RUN_ONCE:
        if is_market_open():
            check_all(symbols)
        else:
            log.info("장 시간이 아니라 종료")
        return

    log.info("PC 모드 시작 — %s %s", MARKET, TICKERS)
    while True:
        try:
            if is_market_open():
                check_all(symbols)
        except Exception as e:
            log.exception("메인 루프 오류 (계속 실행): %s", e)
        time.sleep(CHECK_EVERY_SEC - datetime.now().second % CHECK_EVERY_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("종료")
