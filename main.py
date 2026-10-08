"""
FinxView - Multi Market Trading Chart
Markets: NSE, BSE, US Stocks, Forex, Crypto, Commodities
"""
from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import subprocess
import sys
import tempfile
import json
import shutil

# Needed to actually run arbitrary user-submitted indicator scripts (the
# "Custom Script" engine below). If not installed yet, the rest of the app
# must keep working exactly as before — only that one feature degrades.
try:
    import pandas as pd
    import numpy as np
    _SCRIPT_ENGINE_AVAILABLE = True
except ImportError:
    _SCRIPT_ENGINE_AVAILABLE = False

# Extra libraries scripts commonly want. Each is independently optional —
# one missing/failing library never breaks the others or the core engine
# (which only requires pandas/numpy above). Whatever imports successfully
# here gets handed straight to the script; whatever doesn't just isn't
# available (the script's own `import x` line will fail normally in that
# case, same as running it with plain python).
_SCRIPT_EXTRA_LIBS = {}
for _modname in ("scipy", "sklearn", "statsmodels", "ta", "seaborn", "matplotlib"):
    try:
        _SCRIPT_EXTRA_LIBS[_modname] = __import__(_modname)
    except ImportError:
        pass

# Many indicator scripts are written expecting to pop up a plot window
# (plt.show()) — there's no display on this server, so without this,
# matplotlib would try to open a GUI window and fail. Forcing the headless
# "Agg" backend means those calls just run harmlessly and do nothing
# instead of erroring — this app never renders matplotlib output anyway
# (results go through result/run_smc/events, not a plot).
if "matplotlib" in _SCRIPT_EXTRA_LIBS:
    _SCRIPT_EXTRA_LIBS["matplotlib"].use("Agg")

# Angel One SmartAPI SDK — optional. If it's not installed yet (e.g. .env
# hasn't been set up / `pip install -r requirements.txt` hasn't been re-run
# since it was added), the app must keep working exactly as before on
# yfinance — so this import is allowed to fail silently.
try:
    import importlib.util
    _ANGEL_SDK_AVAILABLE = (
        importlib.util.find_spec("SmartApi") is not None
        and importlib.util.find_spec("pyotp") is not None
    )
except Exception:
    _ANGEL_SDK_AVAILABLE = False

load_dotenv()  # reads SUPABASE_URL / SUPABASE_KEY / ANGEL_* from .env in this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="FinxView", version="3.0")

# ------------------------------------------------------------------
# CORS
# allow_origins=["*"] + allow_credentials=True used to be set together here
# — browsers actually REJECT that exact combination outright (the CORS spec
# forbids a wildcard origin from also being told to send credentials), so
# this was already either silently broken or just hadn't mattered yet
# because nothing here uses cookies. This app is a single browser tab
# talking to its own same-origin backend (index.html is served BY this
# same FastAPI app at "/") — it has no real cross-origin use case at all.
# allow_credentials is now False (nothing here uses cookies to begin with),
# and origins default to just this server's own local address instead of
# "*". If you genuinely need to open the frontend from a different origin
# (e.g. a separate dev server on another port), set FINXVIEW_ALLOWED_ORIGINS
# in .env to a comma-separated list.
# ------------------------------------------------------------------
_default_origins = "http://localhost:8001,http://127.0.0.1:8001"
_allowed_origins = [
    o.strip() for o in os.environ.get("FINXVIEW_ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ------------------------------------------------------------------
# Security headers
# Baseline hardening headers on every response. Doesn't require HTTPS setup
# to add value locally, and costs nothing when you do put this behind
# HTTPS later (a reverse proxy in front of uvicorn, e.g. for LAN/remote
# access — see FINXVIEW_HOST below).
# ------------------------------------------------------------------
@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


# ------------------------------------------------------------------
# Lightweight rate limiting
# No new dependency added for this (same philosophy as the Supabase circuit
# breaker further down — a small in-process guard rather than pulling in a
# library) — a fixed-window counter per (client IP, endpoint group). This is
# NOT meant to survive a restart or work across multiple server processes;
# if you ever run this behind a load balancer with several worker
# processes, move this to Redis (see item 15 of the original brief) so all
# workers share the same counts. For a single local/small-deployment
# instance it's exactly the right amount of protection for what it costs.
# ------------------------------------------------------------------
_rate_limit_buckets = {}  # (ip, group) -> {"window_start": float, "count": int}
_RATE_LIMITS = {
    # group: (max requests, window seconds)
    "run_script": (10, 60),   # spawns a subprocess — the expensive one
    "candles": (120, 60),     # normal chart use is bursty (symbol/timeframe switches)
    "state": (60, 60),
}


def _client_ip(request) -> str:
    # Trust X-Forwarded-For only if you've deliberately put this behind a
    # reverse proxy you control — blindly trusting it otherwise lets a
    # client spoof its way around the limiter by setting the header itself.
    if os.environ.get("FINXVIEW_TRUST_PROXY", "").lower() == "true":
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request, group: str):
    limit, window = _RATE_LIMITS[group]
    key = (_client_ip(request), group)
    now = time.time()
    bucket = _rate_limit_buckets.get(key)
    if bucket is None or (now - bucket["window_start"]) >= window:
        _rate_limit_buckets[key] = {"window_start": now, "count": 1}
        return
    bucket["count"] += 1
    if bucket["count"] > limit:
        raise HTTPException(status_code=429, detail=f"Too many requests — limit is {limit} per {window}s for this endpoint. Try again shortly.")

# ============================================================
# SUPABASE PERSISTENCE
# The server (running on your own laptop, as before) is the only thing
# that talks to Supabase — your Supabase key stays in .env on this
# machine and is never sent to the browser. Workspace data (drawings,
# alerts, favorites, drawing templates, favorites-bar position) is stored
# as a single JSON blob in the "app_state" table, so it survives page
# refresh, closing the app, or restarting the server.
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
STATE_KEY = "workspace"


def supabase_configured():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


# If Supabase is unreachable/misconfigured, every save/load used to wait out
# a full 10s timeout before failing — and since drawing/indicator changes
# trigger a save fairly often, a broken Supabase connection made the whole
# app feel slow, not just workspace sync. This breaker makes that fail fast
# instead: after one failure, stop even attempting the network call for a
# cooldown window, so a real outage costs one slow request, not hundreds.
_supabase_circuit = {"open_until": 0}
_SUPABASE_TIMEOUT = 4            # seconds — was 10
_SUPABASE_CIRCUIT_COOLDOWN = 30  # seconds


def _supabase_circuit_open():
    return time.time() < _supabase_circuit["open_until"]


def _trip_supabase_circuit():
    _supabase_circuit["open_until"] = time.time() + _SUPABASE_CIRCUIT_COOLDOWN


@app.get("/api/state")
def get_state(request: Request):
    _check_rate_limit(request, "state")
    if not supabase_configured():
        raise HTTPException(status_code=503, detail="Supabase not configured — add SUPABASE_URL and SUPABASE_KEY to .env")
    if _supabase_circuit_open():
        raise HTTPException(status_code=503, detail="Supabase temporarily unavailable — retrying automatically, local data is unaffected")
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/app_state",
            params={"key": f"eq.{STATE_KEY}", "select": "value"},
            headers=supabase_headers(),
            timeout=_SUPABASE_TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
        return {"value": rows[0]["value"] if rows else None}
    except Exception as e:
        print(f"[supabase get fail] {str(e)[:200]}")
        _trip_supabase_circuit()
        raise HTTPException(status_code=502, detail="Could not reach Supabase")


@app.post("/api/state")
def set_state(request: Request, payload: dict = Body(...)):
    _check_rate_limit(request, "state")
    if not supabase_configured():
        raise HTTPException(status_code=503, detail="Supabase not configured — add SUPABASE_URL and SUPABASE_KEY to .env")
    if _supabase_circuit_open():
        raise HTTPException(status_code=503, detail="Supabase temporarily unavailable — retrying automatically, local data is unaffected")
    try:
        row = {"key": STATE_KEY, "value": payload.get("value")}
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/app_state",
            headers={**supabase_headers(), "Prefer": "resolution=merge-duplicates"},
            json=row,
            timeout=_SUPABASE_TIMEOUT,
        )
        r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        print(f"[supabase set fail] {str(e)[:200]}")
        _trip_supabase_circuit()
        raise HTTPException(status_code=502, detail="Could not save to Supabase")


# ALL MARKETS - NSE, BSE, US, Forex, Crypto, Commodity
# Real data only (yfinance / Yahoo Finance) — no demo/fallback prices
# ============================================================
SYMBOLS = {
    # ======= NSE INDICES =======
    "^NSEI":      {"name": "NIFTY 50",           "category": "NSE Indices"},
    "^NSEBANK":   {"name": "BANK NIFTY",          "category": "NSE Indices"},
    "^CNXIT":     {"name": "NIFTY IT",            "category": "NSE Indices"},
    "^CNXAUTO":   {"name": "NIFTY AUTO",          "category": "NSE Indices"},
    "^CNXPHARMA": {"name": "NIFTY PHARMA",        "category": "NSE Indices"},
    "^CNXFMCG":   {"name": "NIFTY FMCG",          "category": "NSE Indices"},

    # ======= BSE =======
    "^BSESN":  {"name": "SENSEX",      "category": "BSE"},
    "BSE.NS":  {"name": "BSE Limited", "category": "BSE"},

    # ======= NSE STOCKS - Banking =======
    "PNB.NS":       {"name": "Punjab National Bank",  "category": "NSE Banking"},
    "SBIN.NS":      {"name": "State Bank of India",   "category": "NSE Banking"},
    "HDFCBANK.NS":  {"name": "HDFC Bank",             "category": "NSE Banking"},
    "ICICIBANK.NS": {"name": "ICICI Bank",            "category": "NSE Banking"},
    "AXISBANK.NS":  {"name": "Axis Bank",             "category": "NSE Banking"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank",   "category": "NSE Banking"},
    "INDUSINDBK.NS":{"name": "IndusInd Bank",         "category": "NSE Banking"},
    "IDFCFIRSTB.NS":{"name": "IDFC First Bank",       "category": "NSE Banking"},
    "BANKBARODA.NS":{"name": "Bank of Baroda",        "category": "NSE Banking"},
    "CANBK.NS":     {"name": "Canara Bank",           "category": "NSE Banking"},
    "YESBANK.NS":   {"name": "Yes Bank",              "category": "NSE Banking"},
    "BAJFINANCE.NS":{"name": "Bajaj Finance",         "category": "NSE Banking"},
    "BAJAJFINSV.NS":{"name": "Bajaj Finserv",         "category": "NSE Banking"},

    # ======= NSE STOCKS - IT =======
    "TCS.NS":     {"name": "Tata Consultancy Services", "category": "NSE IT"},
    "INFY.NS":    {"name": "Infosys",                   "category": "NSE IT"},
    "WIPRO.NS":   {"name": "Wipro",                     "category": "NSE IT"},
    "HCLTECH.NS": {"name": "HCL Technologies",          "category": "NSE IT"},
    "TECHM.NS":   {"name": "Tech Mahindra",              "category": "NSE IT"},
    "LTM.NS":    {"name": "LTM (LTIMindtree)",                "category": "NSE IT"},
    "MPHASIS.NS": {"name": "Mphasis",                    "category": "NSE IT"},
    "PERSISTENT.NS": {"name": "Persistent Systems",      "category": "NSE IT"},
    "COFORGE.NS": {"name": "Coforge",                    "category": "NSE IT"},

    # ======= NSE STOCKS - Others =======
    "RELIANCE.NS":  {"name": "Reliance Industries", "category": "NSE Stocks"},
    "TMPV.NS":{"name": "Tata Motors Passenger Vehicles", "category": "NSE Stocks"},
    "TMCV.NS":{"name": "Tata Motors Commercial Vehicles", "category": "NSE Stocks"},
    "MARUTI.NS":    {"name": "Maruti Suzuki",        "category": "NSE Stocks"},
    "ITC.NS":       {"name": "ITC Limited",          "category": "NSE Stocks"},
    "ADANIENT.NS":  {"name": "Adani Enterprises",    "category": "NSE Stocks"},
    "ADANIPORTS.NS":{"name": "Adani Ports",          "category": "NSE Stocks"},
    "SUNPHARMA.NS": {"name": "Sun Pharma",           "category": "NSE Stocks"},
    "DRREDDY.NS":   {"name": "Dr Reddy's Labs",      "category": "NSE Stocks"},
    "CIPLA.NS":     {"name": "Cipla",                "category": "NSE Stocks"},
    "DIVISLAB.NS":  {"name": "Divi's Laboratories",  "category": "NSE Stocks"},
    "APOLLOHOSP.NS":{"name": "Apollo Hospitals",     "category": "NSE Stocks"},
    "LT.NS":        {"name": "Larsen & Toubro",      "category": "NSE Stocks"},
    "ASIANPAINT.NS":{"name": "Asian Paints",         "category": "NSE Stocks"},
    "HINDUNILVR.NS":{"name": "Hindustan Unilever",   "category": "NSE Stocks"},
    "NESTLEIND.NS": {"name": "Nestle India",         "category": "NSE Stocks"},
    "ULTRACEMCO.NS":{"name": "UltraTech Cement",     "category": "NSE Stocks"},
    "SHREECEM.NS":  {"name": "Shree Cement",         "category": "NSE Stocks"},
    "TITAN.NS":     {"name": "Titan Company",        "category": "NSE Stocks"},
    "BHARTIARTL.NS":{"name": "Bharti Airtel",        "category": "NSE Stocks"},
    "POWERGRID.NS": {"name": "Power Grid Corp",      "category": "NSE Stocks"},
    "NTPC.NS":      {"name": "NTPC",                 "category": "NSE Stocks"},
    "ONGC.NS":      {"name": "Oil & Natural Gas Corp","category": "NSE Stocks"},
    "COALINDIA.NS": {"name": "Coal India",           "category": "NSE Stocks"},
    "TATAPOWER.NS": {"name": "Tata Power",           "category": "NSE Stocks"},
    "TATASTEEL.NS": {"name": "Tata Steel",           "category": "NSE Stocks"},
    "JSWSTEEL.NS":  {"name": "JSW Steel",            "category": "NSE Stocks"},
    "HINDALCO.NS":  {"name": "Hindalco Industries",  "category": "NSE Stocks"},
    "VEDL.NS":      {"name": "Vedanta",              "category": "NSE Stocks"},
    "SAIL.NS":      {"name": "Steel Authority of India","category": "NSE Stocks"},
    "BAJAJ-AUTO.NS":{"name": "Bajaj Auto",           "category": "NSE Stocks"},
    "EICHERMOT.NS": {"name": "Eicher Motors",        "category": "NSE Stocks"},
    "HEROMOTOCO.NS":{"name": "Hero MotoCorp",        "category": "NSE Stocks"},
    "M&M.NS":       {"name": "Mahindra & Mahindra",  "category": "NSE Stocks"},
    "GRASIM.NS":    {"name": "Grasim Industries",    "category": "NSE Stocks"},
    "DLF.NS":       {"name": "DLF Limited",          "category": "NSE Stocks"},
    "ETERNAL.NS":    {"name": "Eternal (Zomato)",     "category": "NSE Stocks"},
    "PAYTM.NS":     {"name": "One 97 (Paytm)",       "category": "NSE Stocks"},
    "IRCTC.NS":     {"name": "IRCTC",                "category": "NSE Stocks"},
    "IDEA.NS":      {"name": "Vodafone Idea",        "category": "NSE Stocks"},

    # ======= US STOCKS =======
    "AAPL":  {"name": "Apple Inc",        "category": "US Stocks"},
    "GOOGL": {"name": "Alphabet (Google)","category": "US Stocks"},
    "MSFT":  {"name": "Microsoft",        "category": "US Stocks"},
    "AMZN":  {"name": "Amazon",           "category": "US Stocks"},
    "TSLA":  {"name": "Tesla",            "category": "US Stocks"},
    "META":  {"name": "Meta Platforms",   "category": "US Stocks"},
    "NVDA":  {"name": "NVIDIA",           "category": "US Stocks"},
    "NFLX":  {"name": "Netflix",          "category": "US Stocks"},
    "AMD":   {"name": "Advanced Micro Devices", "category": "US Stocks"},
    "INTC":  {"name": "Intel",            "category": "US Stocks"},
    "ORCL":  {"name": "Oracle",           "category": "US Stocks"},
    "CRM":   {"name": "Salesforce",       "category": "US Stocks"},
    "ADBE":  {"name": "Adobe",            "category": "US Stocks"},
    "PYPL":  {"name": "PayPal",           "category": "US Stocks"},
    "UBER":  {"name": "Uber Technologies","category": "US Stocks"},
    "SHOP":  {"name": "Shopify",          "category": "US Stocks"},
    "PLTR":  {"name": "Palantir Technologies", "category": "US Stocks"},
    "BABA":  {"name": "Alibaba Group",    "category": "US Stocks"},
    "DIS":   {"name": "Walt Disney",      "category": "US Stocks"},
    "KO":    {"name": "Coca-Cola",        "category": "US Stocks"},
    "PEP":   {"name": "PepsiCo",          "category": "US Stocks"},
    "WMT":   {"name": "Walmart",          "category": "US Stocks"},
    "JPM":   {"name": "JPMorgan Chase",   "category": "US Stocks"},
    "V":     {"name": "Visa",             "category": "US Stocks"},
    "MA":    {"name": "Mastercard",       "category": "US Stocks"},
    "BA":    {"name": "Boeing",           "category": "US Stocks"},
    "XOM":   {"name": "ExxonMobil",       "category": "US Stocks"},
    "JNJ":   {"name": "Johnson & Johnson","category": "US Stocks"},
    "PG":    {"name": "Procter & Gamble", "category": "US Stocks"},
    "UNH":   {"name": "UnitedHealth Group","category": "US Stocks"},
    "HD":    {"name": "Home Depot",       "category": "US Stocks"},
    "COST":  {"name": "Costco Wholesale", "category": "US Stocks"},
    "NKE":   {"name": "Nike",             "category": "US Stocks"},
    "SBUX":  {"name": "Starbucks",        "category": "US Stocks"},
    "IBM":   {"name": "IBM",              "category": "US Stocks"},
    "QCOM":  {"name": "Qualcomm",         "category": "US Stocks"},

    # ======= US INDICES =======
    "^GSPC": {"name": "S&P 500",   "category": "US Indices"},
    "^DJI":  {"name": "Dow Jones", "category": "US Indices"},
    "^IXIC": {"name": "NASDAQ",    "category": "US Indices"},

    # ======= CRYPTO =======
    "BTC-USD":  {"name": "Bitcoin",      "category": "Crypto"},
    "ETH-USD":  {"name": "Ethereum",     "category": "Crypto"},
    "BNB-USD":  {"name": "Binance Coin", "category": "Crypto"},
    "SOL-USD":  {"name": "Solana",       "category": "Crypto"},
    "XRP-USD":  {"name": "Ripple",       "category": "Crypto"},
    "ADA-USD":  {"name": "Cardano",      "category": "Crypto"},
    "DOGE-USD": {"name": "Dogecoin",     "category": "Crypto"},
    "DOT-USD":  {"name": "Polkadot",     "category": "Crypto"},
    "LTC-USD":  {"name": "Litecoin",     "category": "Crypto"},
    "TRX-USD":  {"name": "TRON",         "category": "Crypto"},
    "AVAX-USD": {"name": "Avalanche",    "category": "Crypto"},
    "LINK-USD": {"name": "Chainlink",    "category": "Crypto"},
    "UNI-USD":  {"name": "Uniswap",      "category": "Crypto"},
    "ATOM-USD": {"name": "Cosmos",       "category": "Crypto"},
    "XLM-USD":  {"name": "Stellar",      "category": "Crypto"},
    "SHIB-USD": {"name": "Shiba Inu",    "category": "Crypto"},

    # ======= FOREX =======
    "USDINR=X": {"name": "USD / INR", "category": "Forex"},
    "EURINR=X": {"name": "EUR / INR", "category": "Forex"},
    "GBPINR=X": {"name": "GBP / INR", "category": "Forex"},
    "JPYINR=X": {"name": "JPY / INR", "category": "Forex"},
    "EURUSD=X": {"name": "EUR / USD", "category": "Forex"},
    "GBPUSD=X": {"name": "GBP / USD", "category": "Forex"},
    "USDJPY=X": {"name": "USD / JPY", "category": "Forex"},
    "AUDUSD=X": {"name": "AUD / USD", "category": "Forex"},
    "USDCAD=X": {"name": "USD / CAD", "category": "Forex"},
    "USDCHF=X": {"name": "USD / CHF", "category": "Forex"},
    "NZDUSD=X": {"name": "NZD / USD", "category": "Forex"},
    "AUDINR=X": {"name": "AUD / INR", "category": "Forex"},
    "SGDINR=X": {"name": "SGD / INR", "category": "Forex"},

    # ======= COMMODITIES =======
    "GC=F": {"name": "Gold Futures",   "category": "Commodities"},
    "SI=F": {"name": "Silver Futures", "category": "Commodities"},
    "CL=F": {"name": "Crude Oil WTI",  "category": "Commodities"},
    "BZ=F": {"name": "Brent Oil",      "category": "Commodities"},
    "NG=F": {"name": "Natural Gas",    "category": "Commodities"},
    "HG=F": {"name": "Copper",         "category": "Commodities"},
}
TIMEFRAME_MAP = {
    # Yahoo Finance enforces hard limits on how far back intraday data goes,
    # regardless of what's requested — these are the real maximums it allows
    # per interval (not a limit we're imposing):
    #   1m            -> last 7 days only
    #   5m/15m/30m    -> last 60 days
    #   1h (60m)      -> last 730 days (~2 years)
    #   1D/1W/1M      -> full history ("max")
    "1m": {"interval": "1m", "period": "7d"},
    "5m": {"interval": "5m", "period": "60d"},
    "15m": {"interval": "15m", "period": "60d"},
    "30m": {"interval": "30m", "period": "60d"},
    "1h": {"interval": "1h", "period": "730d"},
    "1D": {"interval": "1d", "period": "max"},
    "1W": {"interval": "1wk", "period": "max"},
    "1M": {"interval": "1mo", "period": "max"},
}

# Neither yfinance nor Angel One publish a native 4-hour candle — every
# platform that offers 4h without a native feed builds it by combining
# four 1h candles into one, and that's what fetch_data() does below. It's
# not a key in TIMEFRAME_MAP (there's no direct yfinance/Angel interval for
# it), so it's tracked separately here for the /api/candles validation.
SUPPORTED_TIMEFRAMES = set(TIMEFRAME_MAP.keys()) | {"4h"}


def _resample_hourly_to_4h(candles):
    """Combines every four consecutive 1h candles into one 4h candle.
    Buckets are fixed 4-hour slots aligned to UTC midnight (00:00, 04:00,
    08:00 UTC, etc.) — the same simple, predictable approach used anywhere
    a native 4h feed doesn't exist. candles must be intraday (unix-epoch
    `time`), ascending order."""
    if not candles:
        return []
    FOUR_HOURS = 4 * 3600
    buckets, order = {}, []
    for c in candles:
        bucket_start = (c["time"] // FOUR_HOURS) * FOUR_HOURS
        if bucket_start not in buckets:
            buckets[bucket_start] = dict(c, time=bucket_start)
            order.append(bucket_start)
        else:
            b = buckets[bucket_start]
            b["high"] = max(b["high"], c["high"])
            b["low"] = min(b["low"], c["low"])
            b["close"] = c["close"]  # candles arrive ascending, so this ends up as the bucket's last close
            b["volume"] += c["volume"]
    return [buckets[k] for k in order]


# ============================================================
# ANGEL ONE (SmartAPI) — LIVE DATA FOR INDIVIDUAL NSE STOCKS
#
# Optional and additive: only activates when ANGEL_API_KEY, ANGEL_CLIENT_ID,
# ANGEL_PASSWORD and ANGEL_TOTP_SECRET are all set in .env. When active, it's
# used ONLY for the individual-NSE-equity categories below ("NSE Banking",
# "NSE IT", "NSE Stocks") — Angel One is an India-only broker (no US-stock
# data at all), and NSE/BSE index tokens (NIFTY 50, BANK NIFTY, SENSEX...)
# aren't reliably published in its public instrument list, so indices, BSE,
# US stocks, Forex, Crypto and Commodities all keep using yfinance exactly
# as before — nothing about them changes.
#
# Every failure mode here (not configured, SDK not installed, login fails,
# a symbol isn't found in Angel's instrument list, the request errors out)
# simply returns None, and fetch_data() below falls back to yfinance for
# that symbol — so no stock can ever end up with LESS data than it has
# today because of this integration.
#
# Setup (put these in .env, alongside SUPABASE_URL / SUPABASE_KEY):
#   ANGEL_API_KEY       - from the app you create at smartapi.angelone.in
#   ANGEL_CLIENT_ID     - your Angel One login / client code
#   ANGEL_PASSWORD      - your Angel One login PIN
#   ANGEL_TOTP_SECRET   - the TOTP secret from smartapi.angelone.in/enable-totp
#                         (the same secret an authenticator app would use —
#                         NOT a 6-digit code, the long secret behind the QR)
# ============================================================
ANGEL_API_KEY = os.environ.get("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD = os.environ.get("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET", "")

ANGEL_ELIGIBLE_CATEGORIES = {"NSE Banking", "NSE IT", "NSE Stocks"}

ANGEL_INTERVAL_MAP = {
    "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "1D": "ONE_DAY",
}
# Angel One's own documented max days per single historical request, per
# interval (going over this makes the request fail outright). 1W/1M are
# built by resampling ONE_DAY data ourselves (Angel has no native weekly/
# monthly candle), so they reuse the 1D cap.
ANGEL_MAX_DAYS = {"1m": 30, "5m": 100, "15m": 200, "30m": 200, "1h": 400, "1D": 2000}

_angel_session = {"obj": None, "logged_in_at": 0, "last_login_attempt": 0}
_ANGEL_SESSION_TTL = 6 * 3600  # proactive re-login every 6h; a failed call also forces one
_ANGEL_LOGIN_COOLDOWN = 30     # seconds — never attempt a login more than once per this
                               # window, no matter how many requests want a fresh one at
                               # once. Without this, many NSE stocks failing around the
                               # same time (e.g. because Angel One is already rate-limiting)
                               # each independently retried login immediately, which is
                               # exactly what compounds into "exceeding access rate".

_angel_instruments = {"map": {}, "loaded_at": 0}
_ANGEL_INSTRUMENTS_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
_ANGEL_INSTRUMENTS_TTL = 12 * 3600  # the file itself is only republished ~once a day


def angelone_configured():
    return bool(_ANGEL_SDK_AVAILABLE and ANGEL_API_KEY and ANGEL_CLIENT_ID and ANGEL_PASSWORD and ANGEL_TOTP_SECRET)


def get_angel_session(force_new: bool = False):
    """Logs into Angel One once and reuses the session. force_new=True is
    used after a request fails, to self-heal from a stale/expired token
    without needing to guess its exact lifetime — but never more than once
    per _ANGEL_LOGIN_COOLDOWN, regardless of how many callers ask at once."""
    now = time.time()
    if not force_new and _angel_session["obj"] and (now - _angel_session["logged_in_at"]) < _ANGEL_SESSION_TTL:
        return _angel_session["obj"]

    if (now - _angel_session["last_login_attempt"]) < _ANGEL_LOGIN_COOLDOWN:
        return _angel_session["obj"]  # whatever we already have (possibly None) — don't hammer login again yet

    _angel_session["last_login_attempt"] = now
    try:
        from SmartApi import SmartConnect
        import pyotp

        obj = SmartConnect(api_key=ANGEL_API_KEY)
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        session = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if not session or not session.get("status"):
            print(f"[AngelOne] Login failed: {session}")
            _angel_session["obj"] = None
            return None
        _angel_session["obj"] = obj
        _angel_session["logged_in_at"] = now
        print("[AngelOne] Logged in")
        return obj
    except Exception as e:
        print(f"[AngelOne] Login error: {str(e)[:150]}")
        _angel_session["obj"] = None
        return None


def get_angel_instrument_map():
    """Downloads/caches Angel One's instrument master into a
    {(exch_seg, symbol): token} lookup. Refreshed at most once every 12h —
    the file itself is only republished ~once a day anyway."""
    now = time.time()
    if _angel_instruments["map"] and (now - _angel_instruments["loaded_at"]) < _ANGEL_INSTRUMENTS_TTL:
        return _angel_instruments["map"]
    try:
        r = requests.get(_ANGEL_INSTRUMENTS_URL, timeout=30)
        r.raise_for_status()
        rows = r.json()
        m = {(row.get("exch_seg"), row.get("symbol")): row.get("token") for row in rows}
        _angel_instruments["map"] = m
        _angel_instruments["loaded_at"] = now
        print(f"[AngelOne] Instrument master loaded — {len(m)} instruments")
        return m
    except Exception as e:
        print(f"[AngelOne] Instrument master download failed: {str(e)[:150]}")
        return _angel_instruments["map"]  # serve a stale copy if we have one, else {}


def get_angel_token(base_symbol: str):
    """base_symbol is the plain NSE ticker with '.NS' already stripped,
    e.g. 'RELIANCE', 'BAJAJ-AUTO'. Angel One lists NSE equities with a
    '-EQ' suffix in its instrument master."""
    return get_angel_instrument_map().get(("NSE", f"{base_symbol}-EQ"))


def _resample_daily_to(candles, bucket):
    """Pure-Python weekly/monthly resample from a list of daily candle
    dicts (ascending by date) — deliberately no pandas dependency, to keep
    this feature's footprint small. bucket is 'W' or 'M'."""
    if not candles:
        return []
    buckets, order = {}, []
    for c in candles:
        d = datetime.strptime(c["time"], "%Y-%m-%d")
        key = (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d") if bucket == "W" else d.strftime("%Y-%m-01")
        if key not in buckets:
            buckets[key] = dict(c, time=key)
            order.append(key)
        else:
            b = buckets[key]
            b["high"] = max(b["high"], c["high"])
            b["low"] = min(b["low"], c["low"])
            b["close"] = c["close"]  # candles arrive in ascending date order, so this ends up as the period's last close
            b["volume"] += c["volume"]
    return [buckets[k] for k in order]


# Angel One's own rate limiter rejects bursts ("Access denied because of
# exceeding access rate") under exactly the load this app can generate —
# the sidebar's bulk price refresh alone sends several concurrent
# requests per batch, and the main chart can be fetching at the same
# time. Capping concurrency and spacing consecutive calls apart smooths
# that out. This doesn't guarantee zero rate-limit hits (Angel One's
# actual threshold isn't published), but it meaningfully reduces them —
# and just as importantly, a request paced to avoid the limit entirely
# saves the time a rejected-then-retried one would have wasted anyway.
_angelone_semaphore = threading.Semaphore(2)
_angelone_pacing_lock = threading.Lock()
_angelone_last_call_time = [0.0]
_ANGELONE_MIN_INTERVAL = 0.25  # seconds between consecutive Angel One calls


def _pace_angelone_call():
    with _angelone_pacing_lock:
        elapsed = time.time() - _angelone_last_call_time[0]
        if elapsed < _ANGELONE_MIN_INTERVAL:
            time.sleep(_ANGELONE_MIN_INTERVAL - elapsed)
        _angelone_last_call_time[0] = time.time()


def fetch_angelone(symbol: str, timeframe: str):
    """Live NSE stock data via Angel One SmartAPI. Returns the same candle
    list shape as fetch_yfinance() (list of {time, open, high, low, close,
    volume} dicts), or None for ANY failure — caller falls back to
    yfinance in that case."""
    if not angelone_configured():
        return None
    with _angelone_semaphore:
        _pace_angelone_call()
        return _fetch_angelone_impl(symbol, timeframe)


def _fetch_angelone_impl(symbol: str, timeframe: str):
    try:
        base = symbol[:-3] if symbol.endswith(".NS") else symbol
        token = get_angel_token(base)
        if not token:
            return None

        fetch_tf = "1D" if timeframe in ("1W", "1M") else timeframe
        interval = ANGEL_INTERVAL_MAP.get(fetch_tf)
        if not interval:
            return None

        to_date = datetime.utcnow() + timedelta(hours=5, minutes=30)  # IST, regardless of the
        from_date = to_date - timedelta(days=ANGEL_MAX_DAYS.get(fetch_tf, 30))  # server's own timezone
        candle_params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }

        obj = get_angel_session()
        if not obj:
            return None
        try:
            resp = obj.getCandleData(candle_params)
            if not resp or not resp.get("status"):
                raise ValueError(resp.get("message") if resp else "empty response")
        except Exception:
            # Session may have gone stale — force one fresh login and retry
            # once, rather than guessing exactly when Angel expires tokens.
            obj = get_angel_session(force_new=True)
            if not obj:
                return None
            resp = obj.getCandleData(candle_params)
            if not resp or not resp.get("status"):
                return None

        rows = resp.get("data") or []
        candles = []
        for row in rows:
            # row = [isoTimestamp, open, high, low, close, volume]
            dt = datetime.fromisoformat(row[0])
            o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            v = int(row[5] or 0)
            if any(x <= 0 for x in [o, h, l, c]):
                continue
            time_val = dt.strftime("%Y-%m-%d") if fetch_tf == "1D" else int(dt.timestamp())
            candles.append({"time": time_val, "open": round(o, 4), "high": round(h, 4),
                             "low": round(l, 4), "close": round(c, 4), "volume": v})

        if timeframe in ("1W", "1M"):
            candles = _resample_daily_to(candles, "W" if timeframe == "1W" else "M")

        return candles if len(candles) > 5 else None
    except Exception as e:
        print(f"[AngelOne fail] {symbol} {timeframe}: {str(e)[:150]}")
        return None


def fetch_yfinance(symbol: str, timeframe: str):
    """Fetch real data from Yahoo Finance using yf.download() - avoids Ticker session bugs"""
    try:
        config = TIMEFRAME_MAP[timeframe]

        # yf.download() is more stable than ticker.history() - avoids 'str has no attribute name' bug
        # multi_level_index=False prevents MultiIndex columns in newer yfinance (>=0.2.54)
        try:
            df = yf.download(
                tickers=symbol,
                period=config["period"],
                interval=config["interval"],
                auto_adjust=True,
                prepost=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except TypeError:
            # Older yfinance doesn't support multi_level_index param
            df = yf.download(
                tickers=symbol,
                period=config["period"],
                interval=config["interval"],
                auto_adjust=True,
                prepost=False,
                progress=False,
                threads=False,
            )

        if df is None or df.empty:
            return None

        # Flatten MultiIndex columns if present (happens with single ticker in some yfinance versions)
        if hasattr(df.columns, 'levels') or any(isinstance(c, tuple) for c in df.columns):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return None

        df = df.dropna(subset=required)
        if df.empty:
            return None

        candles = []
        for index, row in df.iterrows():
            if config["interval"] in ["1d", "1wk", "1mo"]:
                time_val = index.strftime("%Y-%m-%d")
            else:
                try:
                    ts = index.timestamp()
                except Exception:
                    ts = int(index.value) // 10**9
                time_val = int(ts)

            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
            c = float(row["Close"])
            try:
                v_raw = row.get("Volume", 0) if hasattr(row, 'get') else (row["Volume"] if "Volume" in row.index else 0)
                v = int(float(v_raw)) if v_raw == v_raw and float(v_raw) > 0 else 0
            except Exception:
                v = 0
            if any(x <= 0 for x in [o, h, l, c]):
                continue
            candles.append({
                "time": time_val,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(c, 4),
                "volume": v,
            })
        return candles if len(candles) > 5 else None
    except Exception as e:
        print(f"[yfinance fail] {symbol}: {str(e)[:120]}")
        return None


# ============================================================
# SHORT-LIVED CANDLE CACHE
# Every /api/candles call used to hit yfinance fresh — a real network round
# trip to Yahoo Finance every single time, including a full "max" history
# re-download for 1D/1W/1M. That's what made switching between timeframes
# (1W -> 1D -> 1H -> 5m -> back to 1D, etc.) feel slow: each switch paid the
# full yfinance latency again even for a timeframe you'd just looked at.
# This cache keeps the last fetch per (symbol, timeframe) in memory for a
# short TTL, so flipping back and forth between the timeframes/symbols
# you're actively working with is instant, while still refreshing often
# enough to stay live. Local single-process app, so a plain dict is enough
# — no need for a real cache server.
# ============================================================
_candle_cache = {}  # (symbol, timeframe) -> {"data": [...], "source": str, "ts": float}
# Matches the frontend's own live-refresh cadence (getRefreshInterval() in
# index.html: 15s for 1m/5m/15m/30m, 30s otherwise) — the cache never holds
# data any longer than the app was already going to wait before refreshing
# it, so this purely removes *redundant* fetches without making anything
# feel less live than it already did.
_INTRADAY_TFS = {"1m", "5m", "15m", "30m"}
_CACHE_TTL_INTRADAY = 15   # seconds
_CACHE_TTL_SLOW = 30       # seconds — covers 1h, 1D, 1W, 1M


def _cache_ttl(timeframe: str) -> int:
    return _CACHE_TTL_INTRADAY if timeframe in _INTRADAY_TFS else _CACHE_TTL_SLOW


def fetch_data(symbol: str, timeframe: str, force_refresh: bool = False):
    """Real data only — no demo/fake fallback: if the real fetch fails,
    callers get an honest 'no data' error instead of synthetic candles.
    Cached briefly per (symbol, timeframe) — see _candle_cache above — so
    re-visiting a timeframe you already loaded doesn't pay the network
    round trip again.

    Individual NSE stocks go through Angel One first when it's configured
    (see ANGEL_ELIGIBLE_CATEGORIES above); everything else — indices, BSE,
    US stocks, Forex, Crypto, Commodities — goes straight to yfinance,
    unchanged from before."""
    key = (symbol, timeframe)
    now = time.time()
    cached = _candle_cache.get(key)
    if not force_refresh and cached and (now - cached["ts"]) < _cache_ttl(timeframe):
        return cached["data"], cached["source"]

    if timeframe == "4h":
        # No native 4h feed anywhere — fetch 1h (through the exact same
        # Angel One / yfinance routing + cache as everything else) and
        # combine every 4 bars into one. Recursing here means 4h gets the
        # same source, fallback behavior, and caching as 1h for free.
        hourly, source = fetch_data(symbol, "1h", force_refresh=force_refresh)
        data = _resample_hourly_to_4h(hourly) if hourly else None
        if data:
            _candle_cache[key] = {"data": data, "source": source, "ts": now}
            return data, source
        return None, None

    category = SYMBOLS.get(symbol, {}).get("category")
    if category in ANGEL_ELIGIBLE_CATEGORIES and angelone_configured():
        print(f"[Fetching] {symbol} {timeframe} - trying Angel One...")
        data = fetch_angelone(symbol, timeframe)
        if data:
            print(f"[OK] Angel One returned {len(data)} candles")
            _candle_cache[key] = {"data": data, "source": "angelone", "ts": now}
            return data, "angelone"
        print(f"[AngelOne] No data for {symbol} {timeframe} — falling back to yfinance")

    print(f"[Fetching] {symbol} {timeframe} - trying yfinance...")
    data = fetch_yfinance(symbol, timeframe)
    if data:
        print(f"[OK] yfinance returned {len(data)} candles")
        _candle_cache[key] = {"data": data, "source": "yfinance", "ts": now}
        return data, "yfinance"

    print(f"[Fail] No real data available for {symbol} {timeframe}")
    # If a fresh fetch fails but we have a (possibly stale) cached copy,
    # serve that rather than a hard failure — matches how the live-refresh
    # timer already tolerates a missed tick.
    if cached:
        return cached["data"], cached["source"]
    return None, None


@app.get("/api/symbols")
def get_symbols():
    grouped = {}
    for symbol, info in SYMBOLS.items():
        cat = info["category"]
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append({"symbol": symbol, "name": info["name"]})
    return {"symbols": grouped}


@app.get("/api/candles")
def get_candles(request: Request, symbol: str = Query(...), timeframe: str = Query("1D")):
    _check_rate_limit(request, "candles")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail="Symbol not found")

    candles, source = fetch_data(symbol, timeframe)
    if not candles:
        raise HTTPException(status_code=404, detail="No data available")

    latest = candles[-1]
    prev = candles[-2] if len(candles) > 1 else latest
    change = latest["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100 if prev["close"] else 0

    return {
        "symbol": symbol,
        "name": SYMBOLS[symbol]["name"],
        "timeframe": timeframe,
        "source": source,
        "candles": candles,
        "latest": {
            "price": latest["close"],
            "change": round(change, 4),
            "change_pct": round(change_pct, 2),
            "high": latest["high"],
            "low": latest["low"],
            "open": latest["open"],
        }
    }


def _batch_yfinance_live_prices(symbols):
    result = {}

    if not symbols:
        return result

    try:
        df = yf.download(
            tickers=symbols,
            period="5d",
            interval="1d",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )

        if df is None or df.empty:
            return result

        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    close_series = df["Close"].dropna()
                else:
                    if symbol not in df.columns.get_level_values(0):
                        continue
                    close_series = df[symbol]["Close"].dropna()

                if close_series.empty:
                    continue

                price = float(close_series.iloc[-1])
                prev = float(close_series.iloc[-2]) if len(close_series) > 1 else price

                change = price - prev
                change_pct = (change / prev) * 100 if prev else 0

                if price < 0.0001:
                    price_out = round(price, 10)
                    change_out = round(change, 10)
                elif price < 0.01:
                    price_out = round(price, 8)
                    change_out = round(change, 8)
                elif price < 1:
                    price_out = round(price, 6)
                    change_out = round(change, 6)
                else:
                    price_out = round(price, 4)
                    change_out = round(change, 4)

                result[symbol] = {
                    "price": price_out,
                    "change": change_out,
                    "change_pct": round(change_pct, 2),
                    "source": "yfinance_batch",
                }

            except Exception as e:
                print(f"[batch yfinance parse fail] {symbol}: {str(e)[:100]}")

    except Exception as e:
        print(f"[batch yfinance fail] {str(e)[:200]}")

    # ---------------------------------------------------------
    # 2B. CRYPTO FALLBACK — COINGECKO
    # Yahoo Finance may not provide some crypto symbols.
    # ---------------------------------------------------------
    crypto_ids = {
        "UNI-USD": "uniswap",
        "SHIB-USD": "shiba-inu",
    }

    missing_crypto = [
        symbol for symbol in crypto_ids
        if symbol in symbols and symbol not in result
    ]

    if missing_crypto:
        try:
            import urllib.parse
            import urllib.request

            ids = ",".join(crypto_ids[s] for s in missing_crypto)
            url = (
                "https://api.coingecko.com/api/v3/simple/price?"
                + urllib.parse.urlencode({
                    "ids": ids,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true"
                })
            )

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FinxView/1.0"}
            )

            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))

            for symbol in missing_crypto:
                coin_id = crypto_ids[symbol]
                item = data.get(coin_id, {})
                price = float(item.get("usd") or 0)
                change_pct = float(item.get("usd_24h_change") or 0)

                if price <= 0:
                    continue

                if price < 0.0001:
                    price_out = round(price, 10)
                elif price < 0.01:
                    price_out = round(price, 8)
                elif price < 1:
                    price_out = round(price, 6)
                else:
                    price_out = round(price, 4)

                change = price * change_pct / 100

                result[symbol] = {
                    "price": price_out,
                    "change": round(change, 10 if price < 0.0001 else 6),
                    "change_pct": round(change_pct, 2),
                    "source": "coingecko",
                }

        except Exception as e:
            print(f"[crypto fallback fail] {str(e)[:200]}")


    return result


_live_prices_cache = None
_live_prices_cache_time = 0.0
LIVE_PRICES_CACHE_TTL = 10

@app.get("/api/live-prices")
def get_live_prices():
    """
    Fast sidebar live-price endpoint.

    NSE stocks:
        Angel One batch LTP.

    Other markets:
        One batched yfinance request instead of one request per symbol.
    """
    global _live_prices_cache, _live_prices_cache_time

    now = time.time()

    if (
        _live_prices_cache is not None
        and now - _live_prices_cache_time < LIVE_PRICES_CACHE_TTL
    ):
        return _live_prices_cache

    result = {}

    # ---------------------------------------------------------
    # 1. ANGEL ONE — BATCH LTP FOR NSE EQUITIES
    # ---------------------------------------------------------
    try:
        if angelone_configured():
            angel = get_angel_session()

            if angel:
                nse_tokens = {}
                token_to_symbol = {}

                for symbol, info in SYMBOLS.items():
                    try:
                        if info.get("category") in ANGEL_ELIGIBLE_CATEGORIES:
                            base = symbol[:-3] if symbol.endswith(".NS") else symbol
                            token = get_angel_token(base)

                            if token:
                                token = str(token)
                                nse_tokens[symbol] = token
                                token_to_symbol[token] = symbol

                    except Exception as e:
                        print(f"[live-price token fail] {symbol}: {str(e)[:100]}")

                if nse_tokens:
                    items = list(nse_tokens.items())

                    # Angel One supports limited tokens per request.
                    # Split NSE symbols into batches of 20.
                    for i in range(0, len(items), 20):
                        batch = items[i:i + 20]

                        batch_symbols = {symbol: token for symbol, token in batch}
                        batch_token_to_symbol = {
                            str(token): symbol
                            for symbol, token in batch
                        }

                        try:
                            response = angel.getMarketData(
                                "FULL",
                                {"NSE": list(batch_symbols.values())}
                            )

                            if response and response.get("status"):
                                fetched = response.get("data", {}).get("fetched", [])

                                for item in fetched:
                                    symbol = batch_token_to_symbol.get(
                                        str(item.get("symbolToken"))
                                    )

                                    if not symbol:
                                        continue

                                    price = float(item.get("ltp") or 0)

                                    if price <= 0:
                                        continue

                                    result[symbol] = {
                                        "price": round(price, 4),
                                        "change": round(float(item.get("netChange") or 0), 4),
                                        "change_pct": round(float(item.get("percentChange") or 0), 2),
                                        "source": "angelone_ltp",
                                    }

                        except Exception as batch_error:
                            print(
                                f"[live-price AngelOne batch fail] "
                                f"{i // 20 + 1}: {str(batch_error)[:150]}"
                            )

    except Exception as e:
        print(f"[live-price AngelOne batch fail] {str(e)[:200]}")

    # ---------------------------------------------------------
    # 2. YFINANCE — ONE BATCH REQUEST FOR ALL OTHER SYMBOLS
    # ---------------------------------------------------------
    remaining = [
        symbol
        for symbol in SYMBOLS
        if symbol not in result
    ]

    result.update(_batch_yfinance_live_prices(remaining))

    response_data = {
        "prices": result,
        "timestamp": time.time()
    }

    _live_prices_cache = response_data
    _live_prices_cache_time = time.time()

    return response_data


@app.get("/api/indicators/sma")
def get_sma(symbol: str = Query(...), timeframe: str = Query("1D"), period: int = Query(20)):
    candles, _ = fetch_data(symbol, timeframe)
    if not candles:
        raise HTTPException(status_code=404, detail="No data available")
    sma_data = []
    closes = [c["close"] for c in candles]
    for i in range(len(candles)):
        if i < period - 1:
            continue
        avg = sum(closes[i - period + 1:i + 1]) / period
        sma_data.append({"time": candles[i]["time"], "value": round(avg, 4)})
    return {"sma": sma_data, "period": period}


@app.get("/api/indicators/ema")
def get_ema(symbol: str = Query(...), timeframe: str = Query("1D"), period: int = Query(20)):
    candles, _ = fetch_data(symbol, timeframe)
    if not candles:
        raise HTTPException(status_code=404, detail="No data available")
    closes = [c["close"] for c in candles]
    ema_data = []
    multiplier = 2 / (period + 1)
    ema = None
    for i, close in enumerate(closes):
        if i < period - 1:
            continue
        if ema is None:
            ema = sum(closes[:period]) / period
        else:
            ema = (close - ema) * multiplier + ema
        ema_data.append({"time": candles[i]["time"], "value": round(ema, 4)})
    return {"ema": ema_data, "period": period}


@app.get("/api/indicators/rsi")
def get_rsi(symbol: str = Query(...), timeframe: str = Query("1D"), period: int = Query(14)):
    candles, _ = fetch_data(symbol, timeframe)
    if not candles:
        raise HTTPException(status_code=404, detail="No data available")
    closes = [c["close"] for c in candles]
    if len(closes) < period + 1:
        return {"rsi": [], "period": period}

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    rsi_data = []
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_data.append({"time": candles[i]["time"], "value": round(rsi, 2)})
    return {"rsi": rsi_data, "period": period}


# ============================================================
# CUSTOM SCRIPT ENGINE — runs real, arbitrary Python indicator code
#
# This actually executes whatever Python the person pastes into the "Add
# Indicator" box (imports, def, classes — all of it), unlike the small
# sma()/ema()/etc. mini-language it falls back to for simple one-liners.
# Any valid Python syntax runs; the practical limits are the library set
# below, a 30s timeout, and needing to produce output in one of the shapes
# further down (a script can run with zero errors and still draw nothing
# if it doesn't produce one of those).
#
# IMPORTANT — this is NOT sandboxed. Executed code has the same access to
# this machine as anything else run.py by hand would. That's an acceptable
# trade-off ONLY because this app runs locally, for one person, on their
# own machine (per the docstring at the top of this file) — never paste
# code you don't trust/didn't write yourself into this box, the same way
# you wouldn't run a random .exe someone sent you.
#
# What the script gets:
#   df  - a pandas DataFrame of the current chart's candles, integer-indexed
#         0..n-1 oldest first (so df.loc[i, 'close'] is bar i, matching how
#         most indicator scripts already index bars by position). Columns
#         are available under BOTH namings: time/open/high/low/close/volume
#         and Date/Open/High/Low/Close/Volume (the capitalized yfinance-
#         style names) — use whichever a given script already expects.
#   pd, np - already imported, ready to use. scipy, sklearn, statsmodels,
#         ta, seaborn, matplotlib are installed (see requirements.txt) and
#         importable with a normal `import x` line inside the script —
#         they don't need to be pre-bound here, pip installing them is
#         what makes the script's own import succeed.
#   print() - captured and returned to the app (shows in the status area),
#         instead of only going to your server's terminal.
#
# What the script needs to produce (any one of):
#   1. A `result` dict: result = {"lines": [
#        {"from_bar": int, "to_bar": int, "price": float, "label": str,
#         "color": "#26a69a" (optional), "dashed": bool (optional)}, ...
#      ]}
#      The simplest, most direct contract — use this for any new script.
#   2. `events = [...]` set directly at the top level, OR returned from a
#      run_smc(df) (optionally run_smc(df, swing_len, ind_swing_len))
#      function — a list of (bar_from, bar_to, level, 'bos'|'choch',
#      'up'|'down') tuples, the shape produced by common structure/SMC-
#      detection scripts. Detected and adapted automatically either way.
#   3. Separate marker lists — e.g. ind_marks/bos_marks/choch_marks — each
#      a list of dicts with at least {"price": float} and a bar-position
#      key ("start"/"from_bar"/"bar", optionally "end"/"to_bar" too, and
#      optionally "type"/"label"/"color"). Detected by shape, not name, so
#      any script using this pattern (whatever it calls the lists) works
#      automatically — all matching lists found get combined and drawn.
# ============================================================
# ------------------------------------------------------------------
# Custom Script execution — SANDBOXED IN A SEPARATE OS PROCESS.
#
# Previously this ran via exec() inside a ThreadPoolExecutor worker, in the
# same process as the whole server. That had three real problems, all of
# which the brief's item 18 explicitly calls out:
#   1. exec()'s default globals include the real __builtins__ — a script
#      could do `import os` / `open()` / `eval()` and touch the filesystem,
#      network, or spawn processes.
#   2. future.result(timeout=...) only stops *waiting* on a stuck thread —
#      Python can't forcibly kill a thread. A `while True: pass` script
#      kept eating a full CPU core forever in the background, and with
#      max_workers=2, two such scripts permanently jammed the feature for
#      every user behind them.
#   3. A crash (sys.exit(), a segfault via ctypes, etc.) happened inside
#      the SAME process as the whole trading app.
#
# Now each script run is its own `python sandbox_runner.py` subprocess:
#   - subprocess.kill() on timeout actually terminates it — no lingering
#     background thread.
#   - restricted builtins + a whitelisted import list (see sandbox_runner.py)
#     block the obvious escapes, as defense-in-depth on top of the process
#     boundary.
#   - on Linux, RLIMIT_AS/RLIMIT_CPU inside the child cap memory/CPU too.
#     `resource` doesn't exist on Windows (this app ships via start.bat on
#     Windows), so on Windows the protection is the process boundary +
#     wall-clock timeout + restricted builtins, not a hard memory ceiling —
#     see sandbox_runner.py's docstring. Not claiming otherwise.
#   - a crash/timeout only takes down that one subprocess.
# Because each run now has its own separate pandas module (separate
# process), the old process-wide monkeypatch lock is no longer needed —
# scripts genuinely run concurrently. A semaphore instead just bounds how
# many script subprocesses can be alive at once, so many simultaneous
# requests can't fork-bomb the host.
# ------------------------------------------------------------------
SCRIPT_TIMEOUT_SECONDS = 30  # wall-clock ceiling for the whole subprocess
SCRIPT_CPU_SECONDS = 25      # slightly under the above — see sandbox_runner.py
SCRIPT_MEM_MB = 512
MAX_CONCURRENT_SCRIPTS = 4
_script_concurrency = threading.Semaphore(MAX_CONCURRENT_SCRIPTS)
_SANDBOX_RUNNER_PATH = os.path.join(BASE_DIR, "sandbox_runner.py")


def _adapt_smc_events(events, candles):
    """Either (bar_from, bar_to, level, 'bos'|'choch', 'up'|'down') tuples,
    or dicts with those fields under any of several common key names, ->
    this app's line format, with bar indices resolved to real candle
    times. A dict with exactly 5 keys unpacks into 5 variables without
    raising an exception — but binds them to the dict's *key names*
    (strings), not its values — so dicts must be detected and handled
    explicitly rather than relying on unpacking to fail loudly."""
    lines = []
    n = len(candles)
    for ev in events:
        try:
            if isinstance(ev, dict):
                bar_from = ev.get("start", ev.get("bar_from", ev.get("from_bar", ev.get("pivot", ev.get("index")))))
                bar_to = ev.get("end", ev.get("bar_to", ev.get("to_bar", ev.get("confirm", ev.get("confirmation", ev.get("index"))))))
                level = ev.get("level", ev.get("price"))
                txt = ev.get("name", ev.get("label", ev.get("type", ev.get("kind", ""))))
                direction_raw = ev.get("direction", "up")
                direction = "up" if str(direction_raw).lower() in ("up", "bullish") else ("down" if str(direction_raw).lower() in ("down", "bearish") else direction_raw)
            else:
                bar_from, bar_to, level, txt, direction = ev
            bar_from = max(0, min(int(bar_from), n - 1))
            bar_to = max(0, min(int(bar_to), n - 1))
            lines.append({
                "time_from": candles[bar_from]["time"],
                "time_to": candles[bar_to]["time"],
                "price": float(level),
                "label": str(txt).upper(),
                "color": "#26a69a" if direction == "up" else "#ef5350",
                "dashed": str(txt).lower() == "choch",
            })
        except Exception:
            continue
    return lines


def _adapt_result_lines(result_lines, candles):
    """result["lines"] (bar-index based, per the documented contract) ->
    real candle times, same as _adapt_smc_events."""
    lines = []
    n = len(candles)
    for l in result_lines:
        try:
            bar_from = max(0, min(int(l["from_bar"]), n - 1))
            bar_to = max(0, min(int(l["to_bar"]), n - 1))
            lines.append({
                "time_from": candles[bar_from]["time"],
                "time_to": candles[bar_to]["time"],
                "price": float(l["price"]),
                "label": str(l.get("label", "")),
                "color": l.get("color") or "#2962ff",
                "dashed": bool(l.get("dashed", False)),
            })
        except Exception:
            continue
    return lines


_MARK_TYPE_COLORS = {"IND": "#ff9800", "BOS": "#2962ff", "CHOCH": "#9c27b0"}
# Short spellings scripts use for the same concepts ("#" = inducement,
# "CH" = CHoCH) — only used to pick the color/dash style; the label drawn on
# the chart is still whatever text the script itself used.
_MARK_TYPE_ALIASES = {"#": "IND", "INDUCEMENT": "IND", "CH": "CHOCH"}

# Note: the structural "does this look like a marks list?" check
# (_looks_like_marks_list) now lives inside sandbox_runner.py, since that's
# where the script's raw namespace is actually inspected (in the isolated
# subprocess). This file only ever sees the already-classified {kind, value}
# result the runner produces.


def _convert_marks_list(marks):
    """Converts a list of marker dicts (start/end/price-or-level/type, or
    the from_bar/to_bar/label spelling) into the from_bar/to_bar/price/
    label/color/dashed shape _adapt_result_lines expects. If a dict
    includes a bullish/bearish or up/down "direction", that's used for
    color when the dict doesn't specify its own color explicitly."""
    out = []
    for m in marks:
        bar_from = m.get("start", m.get("from_bar", m.get("bar")))
        bar_to = m.get("end", m.get("to_bar", bar_from))
        price = m.get("price", m.get("level"))
        label = str(m.get("type", m.get("label", m.get("kind", m.get("name", ""))))).upper()
        key = _MARK_TYPE_ALIASES.get(label, label)
        unconfirmed = label.endswith("?")  # e.g. "BOS?" — level exists but was never broken
        direction = str(m.get("direction", "")).upper()
        if m.get("color"):
            color = m["color"]
        elif direction in ("BULLISH", "UP", "BULL"):
            color = "#26a69a"
        elif direction in ("BEARISH", "DOWN", "BEAR"):
            color = "#ef5350"
        elif unconfirmed:
            color = "#787b86"
        else:
            color = _MARK_TYPE_COLORS.get(key, "#2962ff")
        out.append({
            "from_bar": bar_from, "to_bar": bar_to, "price": price,
            "label": label, "color": color, "dashed": key == "CHOCH" or unconfirmed,
        })
    return out


def _run_user_script(code: str, df):
    """Runs the script in an isolated subprocess (sandbox_runner.py) rather
    than exec()-ing it in this process — see the big comment above this
    function's old location (now the "Custom Script execution" block near
    the top of the file) for exactly what that buys us and what it doesn't.

    Column naming: sandbox_runner.py tries the lowercase convention
    (time/open/high/low/close/volume) first, then — only if that raises —
    retries once against the capitalized yfinance-style convention
    (Date/Open/High/Low/Close/Volume), same as before this change.
    """
    if not _script_concurrency.acquire(timeout=SCRIPT_TIMEOUT_SECONDS):
        raise RuntimeError("Too many Custom Scripts running at once — wait a moment and try again.")

    tmpdir = tempfile.mkdtemp(prefix="finxview_script_")
    try:
        code_path = os.path.join(tmpdir, "script.txt")
        data_path = os.path.join(tmpdir, "data.csv")
        result_path = os.path.join(tmpdir, "result.json")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        df[["time", "open", "high", "low", "close", "volume"]].to_csv(data_path, index=False)

        env = dict(os.environ)
        env["FINXVIEW_SCRIPT_MEM_MB"] = str(SCRIPT_MEM_MB)
        env["FINXVIEW_SCRIPT_CPU_SECONDS"] = str(SCRIPT_CPU_SECONDS)
        # Belt-and-suspenders alongside sandbox_runner.py's own
        # matplotlib.use("Agg", force=True) call — this environment
        # variable makes matplotlib pick the headless backend the moment
        # it's imported, regardless of import order. Without either of
        # these, a script's plt.show() on a machine with a real display
        # (this app's actual Windows deployment, unlike this dev sandbox)
        # opens a real desktop window and blocks the subprocess until a
        # human closes it — indistinguishable from "the script hung."
        env["MPLBACKEND"] = "Agg"

        try:
            proc = subprocess.Popen(
                [sys.executable, _SANDBOX_RUNNER_PATH, code_path, data_path, result_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                cwd=tmpdir,
            )
        except Exception as e:
            raise RuntimeError(f"Could not start the script sandbox: {str(e)[:200]}")

        try:
            stdout, stderr = proc.communicate(timeout=SCRIPT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # reap the process, avoid a zombie
            raise TimeoutError(f"Script timed out after {SCRIPT_TIMEOUT_SECONDS}s — check for an infinite loop")

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "error" in raw:
                raise ValueError(raw["error"] + (f"\n\n--- print() output ---\n{raw.get('printed','')}" if raw.get("printed", "").strip() else ""))
            raw["printed"] = raw.get("printed", "")
            return raw

        # The runner exited without ever writing result.json. Two different
        # cases land here:
        #   - killed by a signal (returncode < 0 on POSIX) with no stderr —
        #     this is what a Linux/macOS CPU-time rlimit (SIGXCPU) or the
        #     RLIMIT_NPROC/RLIMIT_AS limits look like from the parent's side.
        #     From the user's perspective this IS a timeout/runaway-script
        #     case, so report it the same way as the wall-clock timeout
        #     above rather than a generic error.
        #   - anything else (bad argv, import failure inside the runner
        #     itself, etc.) is a genuine infrastructure failure — surface
        #     stderr so it's debuggable.
        if proc.returncode is not None and proc.returncode < 0 and not (stderr or "").strip():
            raise TimeoutError(
                f"Script was terminated after using too much CPU or memory "
                f"(limit: {SCRIPT_CPU_SECONDS}s CPU / {SCRIPT_MEM_MB}MB) — check for an infinite loop or runaway memory use."
            )
        detail = (stderr or stdout or "unknown sandbox failure").strip()[:400]
        raise RuntimeError(f"Script sandbox failed to produce a result: {detail}")
    finally:
        _script_concurrency.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/api/run_script")
def run_script(request: Request, payload: dict = Body(...)):
    _check_rate_limit(request, "run_script")
    if not _SCRIPT_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="pandas/numpy not installed — run: pip install -r requirements.txt")
    if not os.path.exists(_SANDBOX_RUNNER_PATH):
        raise HTTPException(status_code=500, detail="sandbox_runner.py is missing next to main.py — Custom Script cannot run without it")

    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1D")
    code = payload.get("code", "")
    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail="Symbol not found")

    candles, _source = fetch_data(symbol, timeframe)
    if not candles:
        raise HTTPException(status_code=404, detail="No chart data available to run the script against")

    df = pd.DataFrame(candles)  # columns: time, open, high, low, close, volume — index 0..n-1, oldest first

    try:
        raw = _run_user_script(code, df)
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Script error: {str(e)[:500]}")

    if raw["kind"] == "smc_events":
        lines = _adapt_smc_events(raw["value"], candles)
    elif raw["kind"] == "marks":
        lines = _adapt_result_lines(_convert_marks_list(raw["value"]), candles)
    else:
        lines = _adapt_result_lines(raw["value"].get("lines", []), candles)

    return {"lines": lines, "printed": raw.get("printed", "")}


@app.get("/api/info")
def get_info(symbol: str = Query(...)):
    """
    Company/instrument details + recent news for the sidebar "Symbol Info"
    panel (mirrors TradingView's details+news panel). yfinance's .info/.news
    need a live network call to Yahoo and are less reliable than the candle
    endpoint, so this always degrades gracefully instead of erroring out.
    """
    meta = SYMBOLS.get(symbol, {})
    fallback = {
        "symbol": symbol,
        "name": meta.get("name", symbol),
        "category": meta.get("category", ""),
        "exchange": "", "sector": "", "industry": "",
        "marketCap": None, "currency": "",
        "news": [],
        "source": "unavailable",
    }
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        news_raw = []
        try:
            news_raw = t.news or []
        except Exception:
            news_raw = []

        news = []
        for n in news_raw[:6]:
            content = n.get("content", n)  # newer yfinance nests fields under "content"
            title = content.get("title") or n.get("title")
            publisher = (content.get("provider") or {}).get("displayName") or n.get("publisher")
            link = (content.get("clickThroughUrl") or {}).get("url") or n.get("link")
            pub_time = content.get("pubDate") or n.get("providerPublishTime")
            if title:
                news.append({
                    "title": title,
                    "publisher": publisher or "",
                    "link": link or "",
                    "time": pub_time,
                })

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or meta.get("name", symbol),
            "category": meta.get("category", ""),
            "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "marketCap": info.get("marketCap"),
            "currency": info.get("currency", ""),
            "news": news,
            "source": "live" if (info or news) else "unavailable",
        }
    except Exception as e:
        print(f"[info fail] {symbol}: {str(e)[:120]}")
        return fallback


# ============================================================
# Serve single HTML file
# ============================================================
@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/favicon.ico")
def favicon():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 binds to EVERY network interface, not just this machine —
    # since nothing in this app has a login, that used to mean anyone else
    # on the same WiFi/LAN (or reachable via port-forwarding) could open
    # your /api/state (drawings/alerts), pull real market data through your
    # Angel One session, and submit Custom Scripts through your server,
    # all with zero authentication. Defaulting to 127.0.0.1 makes this
    # actually match what the comments elsewhere in this file already
    # assume ("runs on your own laptop", "never sent to the browser" etc.).
    # If you genuinely want LAN access (e.g. checking charts from your
    # phone on the same WiFi), set FINXVIEW_HOST=0.0.0.0 in .env — that's a
    # deliberate choice you're opting into, not the default.
    host = os.environ.get("FINXVIEW_HOST", "127.0.0.1")
    port = int(os.environ.get("FINXVIEW_PORT", "8001"))
    print("\n" + "=" * 60)
    print("  FinxView - Multi Market Chart")
    print(f"  Browser: http://localhost:{port}")
    print("  Markets: NSE, BSE, US Stocks, Forex, Crypto, Commodity")
    if host != "127.0.0.1":
        print(f"  [!] Bound to {host} — reachable from other devices on your network, not just this machine.")
    print("  Stop: Ctrl+C")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
