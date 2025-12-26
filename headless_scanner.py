import logging
import asyncio
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import nest_asyncio
import streamlit as st
import time
import os
import gc
import threading
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from telegram import (
    Update, 
    ReplyKeyboardMarkup, 
    KeyboardButton, 
    constants
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    PicklePersistence,
    Application
)
import telegram.error

# --- CONFIGURATION ---
nest_asyncio.apply()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. LOAD SECRETS (ROBUST VERSION)
# ==========================================
try:
    TG_TOKEN = st.secrets["TG_TOKEN"].strip()
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "").strip()
    print(f"✅ Loaded Token: {TG_TOKEN[:5]}... | Admin ID: {ADMIN_ID}")
except Exception as e:
    st.error(f"❌ Secret Error: {e}")
    st.stop()

# 2. GLOBAL SETTINGS
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14
SMA_MAJ = 200

DEFAULT_PARAMS = {
    'risk_usd': 50.0,
    'min_rr': 1.25,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
    'auto_scan': False, # NEW PARAMETER
}

# ==========================================
# 3. CORE LOGIC (MATH & DATA)
# ==========================================

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: return []

def format_market_cap(val):
    if not val or pd.isna(val): return "N/A"
    try:
        if val >= 1e12: return f"{val/1e12:.2f}T"
        if val >= 1e9: return f"{val/1e9:.2f}B"
        if val >= 1e6: return f"{val/1e6:.2f}M"
        return str(val)
    except: return "N/A"

def get_extended_info(ticker):
    try:
        t = yf.Ticker(ticker)
        try: mc = t.fast_info['market_cap']
        except: mc = None
        try:
            i = t.info
            pe = i.get('trailingPE') or i.get('forwardPE')
        except: pe = None
        pe_str = f"{pe:.2f}" if pe else "N/A"
        mc_str = format_market_cap(mc)
        return {"mc": mc_str, "pe": pe_str}
    except:
        return {"mc": "N/A", "pe": "N/A"}

# --- INDICATORS ---
def calc_sma(s, l): return s.rolling(l).mean()
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()
def calc_macd(s, f=12, sl=26, sig=9):
    fast = s.ewm(span=f, adjust=False).mean()
    slow = s.ewm(span=sl, adjust=False).mean()
    macd = fast - slow
    return macd - macd.ewm(span=sig, adjust=False).mean()

def calc_adx_pine(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1); down = l.shift(1) - l
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    def rma(s, len): return s.ewm(alpha=1/len, adjust=False).mean()
    tr_s = rma(tr, length).replace(0, np.nan)
    p_di = 100 * (rma(pd.Series(p_dm, index=df.index), length) / tr_s)
    m_di = 100 * (rma(pd.Series(m_dm, index=df.index), length) / tr_s)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
    return rma(dx, length), p_di, m_di

def calc_atr(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# --- STRATEGY ---
def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    n = len(df)
    c_a, h_a, l_a = df['Close'].values, df['High'].values, df['Low'].values
    
    seq_st = np.zeros(n, dtype=int)
    crit_lvl = np.full(n, np.nan)
    res_peak = np.full(n, np.nan)
    res_struct = np.zeros(n, dtype=bool)
    
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]; s_l = l_a[0]
    
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h 
        prev_sl = s_l 
        
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1: 
                final_high = max(prev_sh, h)
                is_hh = True if np.isnan(last_pk) else (final_high > last_pk)
                pk_hh = is_hh
                last_pk = final_high 
                
                s_state = -1
                s_h = h; s_l = l
                s_crit = h
            else: 
                final_low = min(prev_sl, l)
                is_hl = True if np.isnan(last_tr) else (final_low > last_tr)
                tr_hl = is_hl
                last_tr = final_low
                
                s_state = 1
                s_h = h; s_l = l
                s_crit = l
        else:
            s_state = prev_st
            if s_state == 1:
                if h >= s_h: s_h = h
                if h >= prev_sh: s_crit = l
                else: s_crit = prev_cr
            elif s_state == -1:
                if l <= s_l: s_l = l
                if l <= prev_sl: s_crit = h
                else: s_crit = prev_cr
            else:
                if c > prev_sh: 
                    s_state = 1; s_crit = l
                elif c < prev_sl: 
                    s_state = -1; s_crit = h
                else:
                    s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        
        seq_st[i] = s_state
        crit_lvl[i] = s_crit
        res_peak[i] = last_pk
        res_struct[i] = (pk_hh and tr_hl)

    adx_str = adx >= adx_thr
    bull = (adx_str & (p_di > m_di)) & ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & (efi > 0)
    bear = (adx_str & (m_di > p_di)) & ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & (efi < 0)
    t_st = np.zeros(n, dtype=int)
    t_st[bull] = 1; t_st[bear] = -1
    
    df['Seq'] = seq_st; df['Crit'] = crit_lvl; df['Peak'] = res_peak; df['Struct'] = res_struct; df['Trend'] = t_st; df['ATR'] = atr
    return df

def analyze_trade(df, idx):
    r = df.iloc[idx]
    price = r['Close']; tp = r['Peak']; crit = r['Crit']; atr = r['ATR']
    sma = r['SMA']
    errs = []
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(sma) or price <= sma: errs.append("MA❌")
    if r['Trend'] == -1: errs.append("Trend❌")
    if not r['Struct']: errs.append("Struct❌")
    if np.isnan(tp) or np.isnan(crit): errs.append("NO DATA")

    sl_struct = crit
    sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr) if not np.isnan(sl_struct) else sl_atr
    
    risk = price - final_sl
    reward = tp - price
    
    rr = reward / risk if risk > 0 else 0
    
    data = {"P": price, "TP": tp, "SL": final_sl, "RR": rr, "ATR": atr, "Crit": crit,
            "Seq": r['Seq'], "Trend": r['Trend'], "SMA": sma, "Struct": r['Struct'], "Close": price}
    
    valid = len(errs) == 0
    return valid, data, errs

# ==========================================
# 4. UI: DASHBOARD STYLE (PREMIUM + DYNAMIC SMA)
# ==========================================
def format_dashboard_card(ticker, d, shares, is_new, info, p_risk, sma_len):
    tv_ticker = ticker.replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
    
    pe_str = str(info.get('pe', 'N/A'))
    mc_str = str(info.get('mc', 'N/A'))
    atr_pct = (d['ATR'] / d['Close']) * 100
    
    trend_emo = "🟢" if d['Trend'] == 1 else ("🔴" if d['Trend'] == -1 else "🟡")
    seq_emo = "🟢" if d['Seq'] == 1 else ("🔴" if d['Seq'] == -1 else "🟡")
    ma_emo = "🟢" if d['Close'] > d['SMA'] else "🔴"
    
    cond_seq = d['Seq'] == 1
    cond_ma = d['Close'] > d['SMA']
    cond_trend = d['Trend'] != -1
    cond_struct = d.get('Struct', False)
    
    is_valid_setup = cond_seq and cond_ma and cond_trend and cond_struct
    risk = d['P'] - d['SL']
    reward = d['TP'] - d['P']
    is_valid_math = risk > 0 and reward > 0

    header = f"<b><a href='{tv_link}'>{ticker}</a></b>  ${d['P']:.2f}\n"
    
    context_block = (
        f"MC: {mc_str} | P/E: {pe_str}\n"
        f"ATR: ${d['ATR']:.2f} ({atr_pct:.2f}%)\n"
        f"Trend {trend_emo}  Seq {seq_emo}  MA{sma_len} {ma_emo}\n"
    )

    if is_valid_setup and is_valid_math:
        status_icon = "🆕" if is_new else "♻️"
        profit = reward * shares
        loss = risk * shares
        rr_str = f"{d['RR']:.2f}"
        total_val = shares * d['P']

        html = (
            f"{status_icon} {header}"
            f"Size: {shares} shares (${total_val:,.0f})\n"
            f"{context_block}"
            f"🛑 SL: {d['SL']:.2f}  (-${loss:.0f})\n"
            f"🎯 TP: {d['TP']:.2f}  (+${profit:.0f})\n"
            f"⚖️ Risk/Reward: {rr_str}"
        )
    else:
        reasons = []
        if not cond_seq: reasons.append("Seq❌")
        if not cond_ma: reasons.append("MA❌")
        if not cond_trend: reasons.append("Trend❌")
        if not cond_struct: reasons.append("Struct❌")
        if risk <= 0: reasons.append("❌RR NEGATIVE")
        elif reward <= 0: reasons.append("❌ABOVE HH")

        fail_str = " ".join(reasons) if reasons else "UNKNOWN ERROR"
        html = f"⛔ {header}{context_block}<b>NO SETUP:</b> {fail_str}"
    
    return html

# ==========================================
# 5. SCANNING PROCESS (WITH CONFIG BAR)
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_mode=False):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # --- MEMORY INIT ---
    if 'signal_memory' not in context.user_data:
        context.user_data['signal_memory'] = {}
    
    # Clean memory if new day (NY Time)
    ny_now = datetime.datetime.now(pytz.timezone('US/Eastern')).date()
    # Check if we stored date last time, if not, reset.
    last_date = context.user_data.get('last_mem_date')
    if last_date != str(ny_now):
        context.user_data['signal_memory'] = {}
        context.user_data['last_mem_date'] = str(ny_now)

    # VISUAL CONFIGURATION
    config_display = (
        f"⚙️ <b>Active Settings:</b>\n"
        f"Risk ${p['risk_usd']:.0f} | RR {p['min_rr']} | SMA {p['sma']} | ATR {p['max_atr']}%\n"
        f"TF: {p['tf']} | New Only: {'✅' if p['new_only'] else '❌'}"
    )

    if manual_mode:
        status_msg = await context.bot.send_message(
            chat_id=chat_id, 
            text=f"🔎 <b>Scanning {len(tickers)} tickers...</b>\n\n{config_display}", 
            parse_mode='HTML'
        )
    
    results_found = 0
    total = len(tickers)
    
    for i, t in enumerate(tickers):
        # Stop check
        if not context.user_data.get('scanning', False):
            if manual_mode: await context.bot.send_message(chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break
            
        # UI Update (Manual only)
        if manual_mode and (i % 10 == 0 or i == total - 1):
            try:
                pct = int((i + 1) / total * 10)
                bar = "█" * pct + "░" * (10 - pct)
                percent_num = int((i+1)/total*100)
                await status_msg.edit_text(
                    f"<b>SCAN:</b> {i+1}/{total} ({percent_num}%)\n"
                    f"[{bar}]\n"
                    f"👉 <i>Checking: {t}</i>\n\n"
                    f"{config_display}", 
                    parse_mode='HTML'
                )
            except: pass
            
        if i % 50 == 0: gc.collect()
        
        try:
            await asyncio.sleep(0.01)
            inter = "1d" if p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if p['tf'] == "Daily" else "5y"
            
            df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            if len(df) < p['sma'] + 5:
                if manual_mode: await context.bot.send_message(chat_id, f"⚠️ <b>{t}</b>: Not enough data", parse_mode='HTML')
                continue
            
            df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            valid, d, errs = analyze_trade(df, -1)
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            show_card = False
            
            if manual_mode: 
                show_card = True
            elif valid:
                # MEMORY CHECK FOR AUTO SCAN
                # If not manual mode, check if we already showed this ticker today
                if not manual_mode:
                    if t in context.user_data['signal_memory']:
                        show_card = False # Already shown today
                    else:
                        # Logic filters
                        if p['new_only'] and not is_new: show_card = False
                        elif d['RR'] >= p['min_rr'] and (d['ATR']/d['P'])*100 <= p['max_atr']:
                            risk_per_share = d['P'] - d['SL']
                            if risk_per_share > 0:
                                shares = int(p['risk_usd'] / risk_per_share)
                                if shares >= 1: show_card = True
                else:
                    # Manual mode standard filters
                    if p['new_only'] and not is_new: show_card = False
                    elif d['RR'] >= p['min_rr'] and (d['ATR']/d['P'])*100 <= p['max_atr']:
                        risk_per_share = d['P'] - d['SL']
                        if risk_per_share > 0:
                            shares = int(p['risk_usd'] / risk_per_share)
                            if shares >= 1: show_card = True
            
            if show_card:
                # Add to memory if it's a valid signal (Auto or Manual doesn't matter, we remember seeing it)
                context.user_data['signal_memory'][t] = True
                
                info = get_extended_info(t)
                risk_per_share = d['P'] - d['SL']
                shares = int(p['risk_usd'] / risk_per_share) if risk_per_share > 0 else 0
                
                card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'], p['sma'])
                
                await context.bot.send_message(chat_id=chat_id, text=card, parse_mode='HTML', disable_web_page_preview=True)
                results_found += 1
                
        except Exception as e:
            if manual_mode: await context.bot.send_message(chat_id, f"⚠️ <b>{t} Error:</b> {str(e)}", parse_mode='HTML')
            continue
    
    if manual_mode:       
        await context.bot.send_message(chat_id=chat_id, text=f"🏁 <b>SCAN COMPLETE</b>\n✅ Found: {results_found}", parse_mode='HTML')
        context.user_data['scanning'] = False

# ==========================================
# 6. BOT HANDLERS & HELPERS
# ==========================================
def get_allowed_users():
    allowed = {ADMIN_ID}
    if GITHUB_USERS_URL:
        try:
            response = requests.get(GITHUB_USERS_URL, timeout=3)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    clean = line.split('#')[0].strip()
                    if clean.isdigit(): allowed.add(int(clean))
        except Exception as e: print(f"⚠️ Error fetching whitelist: {e}")
    return allowed

async def check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in get_allowed_users(): 
        try: 
            msg = (
                f"🛑 <b>Authorization Required</b>\n\n"
                f"👋 <b>Welcome!</b> This is a private quantitative scanner.\n"
                f"To get access, you need to be approved by the administrator.\n\n"
                f"📩 Please send your ID number to <b>@Vova_Skl</b>:\n\n"
                f"🆔 <b>Your ID:</b> <code>{user_id}</code>\n"
                f"<i>(Click the number to copy)</i>"
            )
            await update.message.reply_html(msg)
        except: pass
        return False
    
    if 'active_users' not in context.bot_data: context.bot_data['active_users'] = set()
    context.bot_data['active_users'].add(user_id)
    return True

async def safe_get_params(context):
    # Ensure all keys exist
    if 'params' not in context.user_data: 
        context.user_data['params'] = DEFAULT_PARAMS.copy()
    
    p = context.user_data['params']
    # Merge with default to ensure no missing keys after update
    for k, v in DEFAULT_PARAMS.items():
        if k not in p:
            p[k] = v
            
    return p

# --- KEYBOARDS ---
def get_main_keyboard(p):
    risk = f"💸 Risk: ${p['risk_usd']:.0f}"
    rr = f"⚖️ RR: {p['min_rr']}"
    atr = f"📊 ATR Max: {p['max_atr']}%"
    sma = f"📈 SMA: {p['sma']}"
    tf = f"⏳ TIMEFRAME: {p['tf'][0]}"
    new = f"Only New {'✅' if p['new_only'] else '❌'}"
    auto = f"Auto-Scan {'🟢' if p.get('auto_scan') else '🔴'}"
    
    return ReplyKeyboardMarkup([
        [KeyboardButton(risk), KeyboardButton(rr)],
        [KeyboardButton(atr), KeyboardButton(sma)],
        [KeyboardButton(tf), KeyboardButton(new)], 
        [KeyboardButton(auto)], # NEW BUTTON
        [KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP SCAN")],
        [KeyboardButton("ℹ️ HELP / INFO")]
    ], resize_keyboard=True)

def get_sma_keyboard():
    return ReplyKeyboardMarkup([
        [KeyboardButton("SMA 50"), KeyboardButton("SMA 100"), KeyboardButton("SMA 150"), KeyboardButton("SMA 200")], 
        [KeyboardButton("🔙 Back")]
    ], resize_keyboard=True)

def get_tf_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton("Daily (D)"), KeyboardButton("Weekly (W)")], [KeyboardButton("🔙 Back")]], resize_keyboard=True)

# --- COMMANDS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    try:
        p = await safe_get_params(context)
    except:
        p = DEFAULT_PARAMS.copy()
        context.user_data['params'] = p

    context.user_data['input_mode'] = None
    
    # Deep Linking
    if context.args and context.args[0] == 'autoscan':
        await update.message.reply_text("🚀 <b>Auto-starting Scan...</b>", parse_mode='HTML')
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=False))
        return

    user_name = update.effective_user.first_name
    
    welcome_text = f"""👋 <b>Welcome, {user_name}!</b>

🤖 <b>I am the Vova Sequence Screener.</b>
I am a quantitative tool designed to automate the search for high-probability setups in the S&P 500 based on Market Structure.

<b>🧩 STRATEGY DECODED:</b>

<b>1. 📈 Trend Filter (SMA):</b> 
I only trigger signals when the price is ABOVE the <b>SMA {p['sma']}</b>.
<i>(You can change this to SMA 50, 100, 150, or 200 in the menu to suit your style).</i>

<b>2. 🚀 Momentum (Elder Impulse):</b> 
I use the <b>Elder Impulse System</b> (EMA 13/26 + MACD Histogram). I require the bars to be <b>Green</b> (Bullish) at the moment of the signal to ensure we are entering with force.

<b>3. 💪 Trend Strength (ADX):</b> 
I require <b>ADX > {ADX_T}</b>. This filters out "choppy" or ranging markets and ensures the trend has real strength.

<b>4. 🧱 Market Structure:</b> 
I look for a specific "Break of Structure" (BoS). A New Higher High following a valid Higher Low.

<b>🛡️ RISK MANAGEMENT:</b>
• <b>ATR Filter:</b> Rejects stocks with volatility > {p['max_atr']}%.
• <b>Smart Stops:</b> SL is calculated based on Structure or ATR.
• <b>R/R Ratio:</b> Trades below <b>{p['min_rr']}R</b> are skipped.

⚠️ <b>LEGAL DISCLAIMER:</b>
This bot is a <b>screener</b>, not a financial advisor. 
1. <b>No Advice:</b> These signals are mathematical calculations, not buy recommendations.
2. <b>Your Risk:</b> Trading stocks involves high risk. You can lose your entire investment.
3. <b>Verification:</b> Always verify chart patterns manually before trading.

<i>👇 Use the menu below to configure parameters and Start Scan.</i>"""

    await update.message.reply_html(welcome_text, reply_markup=get_main_keyboard(p))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    allowed = get_allowed_users()
    active = context.bot_data.get('active_users', set())
    msg = (f"📊 <b>BOT STATISTICS</b>\n━━━━━━━━━━━━━━━━━━\n"
           f"✅ <b>Approved:</b> {len(allowed)}\n<code>{', '.join(map(str, allowed))}</code>\n\n"
           f"👥 <b>Active:</b> {len(active)}\n<code>{', '.join(map(str, active))}</code>")
    await update.message.reply_html(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    text = update.message.text
    p = await safe_get_params(context)
    
    # --- NAVIGATION ---
    if text == "🔙 Back":
        context.user_data['input_mode'] = None
        await update.message.reply_text("🔙 Main Menu", reply_markup=get_main_keyboard(p))
        return

    # --- ACTIONS ---
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'): return await update.message.reply_text("⚠️ Already running!")
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=True)) # MANUAL MODE = TRUE
        return
    elif text == "⏹ STOP SCAN":
        context.user_data['scanning'] = False
        p['auto_scan'] = False # Stop auto too
        context.user_data['params'] = p # Save stop
        return await update.message.reply_text("🛑 Stopping all scans...", reply_markup=get_main_keyboard(p))
    
    # --- AUTO SCAN TOGGLE ---
    elif "Auto-Scan" in text:
        p['auto_scan'] = not p.get('auto_scan', False)
        status = "ON 🟢" if p['auto_scan'] else "OFF 🔴"
        context.user_data['scanning'] = True # Enable master switch if Auto ON
        
        # Save persistence immediately
        context.user_data['params'] = p
        await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
        
        await update.message.reply_text(f"🔄 Auto-Scan is now {status}", reply_markup=get_main_keyboard(p))
        return

    # --- HELP SECTION ---
    elif text == "ℹ️ HELP / INFO":
        help_text = (
            "<b>📚 TECHNICAL MANUAL & SETTINGS</b>\n\n"
            "<b>📈 SMA (Trend Filter):</b>\n"
            "The bot ignores any stock trading below this line.\n"
            "• <b>SMA 200:</b> Long-term (Conservative).\n"
            "• <b>SMA 50:</b> Medium-term (Aggressive).\n"
            "<i>Use the menu buttons to switch.</i>\n\n"
            "<b>⏳ TIMEFRAME:</b>\n"
            "• <b>Daily (D):</b> Scans 2 years of history.\n"
            "• <b>Weekly (W):</b> Scans 5 years of history.\n"
            "<i>Note: Changing timeframe recalculates ALL indicators (SMA, ATR, Highs/Lows).</i>\n\n"
            "<b>⚙️ PARAMETERS:</b>\n"
            "• <b>Risk ($):</b> The dollar amount you are willing to lose per trade. Used to calculate Share Size.\n"
            "• <b>RR (Reward/Risk):</b> Minimum ratio required. If a setup offers 1.1R but you want 1.5R, it is skipped.\n"
            "• <b>ATR Max (%):</b> Volatility cap. Prevents trading stocks that move too violently (e.g. >5% per day).\n\n"
            "<b>🔍 Manual Scan:</b>\n"
            "Type tickers (e.g. <code>NVDA, TSLA</code>) to scan them immediately.\n\n"
            "⚠️ <b>IMPORTANT DISCLAIMER:</b>\n"
            "This software is provided 'as is' for informational purposes only. It is not financial advice. "
            "The user assumes full responsibility for any trading decisions and results. "
            "The algorithms filter data but cannot predict future market movements with certainty."
        )
        return await update.message.reply_html(help_text)
    
    # --- SUB-MENUS ---
    elif "SMA:" in text:
        context.user_data['input_mode'] = "sma_select"
        await update.message.reply_text("Select SMA Length:", reply_markup=get_sma_keyboard())
        return
    elif "TIMEFRAME:" in text:
        context.user_data['input_mode'] = "tf_select"
        await update.message.reply_text("Select Timeframe:", reply_markup=get_tf_keyboard())
        return

    # --- SELECTION LOGIC ---
    if context.user_data.get('input_mode') == "sma_select":
        if text in ["SMA 50", "SMA 100", "SMA 150", "SMA 200"]:
            p['sma'] = int(text.split()[1])
            context.user_data['input_mode'] = None
            
            # Save persistence
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            
            await update.message.reply_text(f"✅ SMA set to {p['sma']}", reply_markup=get_main_keyboard(p))
        return
    if context.user_data.get('input_mode') == "tf_select":
        if "Daily" in text: p['tf'] = "Daily"
        elif "Weekly" in text: p['tf'] = "Weekly"
        context.user_data['input_mode'] = None
        
        # Save persistence
        context.user_data['params'] = p
        await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
        
        await update.message.reply_text(f"✅ Timeframe set to {p['tf']}", reply_markup=get_main_keyboard(p))
        return

    # --- TOGGLES & INPUTS ---
    if "Only New" in text: 
        p['new_only'] = not p['new_only']
        status = "ENABLED" if p['new_only'] else "DISABLED"
        
        # Save persistence
        context.user_data['params'] = p
        await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
        
        await update.message.reply_text(f"✅ Only New Signals: {status}", reply_markup=get_main_keyboard(p))
        return

    if "Risk:" in text:
        context.user_data['input_mode'] = "risk"
        return await update.message.reply_text("✏️ Enter Risk Amount ($):")
    elif "RR:" in text:
        context.user_data['input_mode'] = "rr"
        return await update.message.reply_text("✏️ Enter Min R/R (e.g. 1.5):")
    elif "ATR Max:" in text:
        context.user_data['input_mode'] = "atr"
        return await update.message.reply_text("✏️ Enter Max ATR % (e.g. 5.0):")

    mode = context.user_data.get('input_mode')
    if mode == "risk":
        try: 
            val = float(text)
            if val < 1: raise ValueError
            p['risk_usd'] = val
            context.user_data['input_mode'] = None
            
            # Save
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            
            await update.message.reply_text(f"✅ Risk updated to ${val}", reply_markup=get_main_keyboard(p))
        except: await update.message.reply_text("❌ Invalid amount.")
        return
    elif mode == "rr":
        try:
            val = float(text)
            if val < 0.1: raise ValueError
            p['min_rr'] = val
            context.user_data['input_mode'] = None
            
            # Save
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            
            await update.message.reply_text(f"✅ Min RR updated to {val}", reply_markup=get_main_keyboard(p))
        except: await update.message.reply_text("❌ Invalid number.")
        return
    elif mode == "atr":
        try:
            val = float(text)
            if val < 0.1: raise ValueError
            p['max_atr'] = val
            context.user_data['input_mode'] = None
            
            # Save
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            
            await update.message.reply_text(f"✅ Max ATR updated to {val}%", reply_markup=get_main_keyboard(p))
        except: await update.message.reply_text("❌ Invalid number.")
        return

    # --- MANUAL ENTRY ---
    if "," in text or (text.isalpha() and len(text) < 6):
        ts = [x.strip().upper() for x in text.split(",") if x.strip()]
        if ts:
            context.user_data['scanning'] = True
            await context.bot.send_message(update.effective_chat.id, f"🔎 Diagnosing: {ts}")
            await run_scan_process(update, context, p, ts, manual_mode=True)
        return

    # --- DEFAULT RESPONSE (SHOW CONFIG) ---
    context.user_data['params'] = p
    config_str = f"⚙️ <b>Active Config:</b> {p['tf']} | SMA {p['sma']} | Risk ${p['risk_usd']}"
    await update.message.reply_html(config_str, reply_markup=get_main_keyboard(p))

# ==========================================
# 7. ARCHITECTURE: SINGLETON BOT + AUTO-SCHEDULER
# ==========================================

async def auto_scan_scheduler(app):
    """
    Background Task: 
    1. Checks time (US Market Hours: 09:35 - 15:35 ET)
    2. Iterates over ALL users with persistence data
    3. If user has 'auto_scan': True -> Runs scan for them
    """
    while True:
        try:
            await asyncio.sleep(60) # Check every minute
            
            # 1. Market Time Check
            ny_tz = pytz.timezone('US/Eastern')
            now = datetime.datetime.now(ny_tz)
            
            # Market hours: 09:30 - 16:00. We scan 09:35 - 15:35 to be safe
            is_market_open = (
                now.weekday() < 5 and # Mon-Fri
                (
                    (now.hour == 9 and now.minute >= 35) or
                    (now.hour >= 10 and now.hour < 15) or
                    (now.hour == 15 and now.minute <= 35)
                )
            )
            
            # Run scan ONLY at minute 05 of the hour (Once per hour)
            if is_market_open and now.minute == 5:
                print(f"⏰ Auto-Scan Triggered at {now.strftime('%H:%M')} ET")
                
                # Iterate all users in persistence
                if not app.user_data: continue
                
                tickers = get_sp500_tickers()
                
                for user_id, data in app.user_data.items():
                    p = data.get('params', DEFAULT_PARAMS.copy())
                    
                    if p.get('auto_scan', False):
                        # Create a fake update/context to reuse run_scan_process
                        # We need to construct a minimal context
                        
                        # Manually check limits
                        data['scanning'] = True
                        
                        # We must create a task for each user so one doesn't block others
                        # But we reuse the logic function carefully
                        class MockUpdate:
                            class EffectiveUser:
                                id = user_id
                            effective_user = EffectiveUser()
                            class EffectiveChat:
                                id = user_id
                            effective_chat = EffectiveChat()
                        
                        class MockContext:
                            bot = app.bot
                            user_data = data
                        
                        # Run async scan
                        await run_scan_process(MockUpdate(), MockContext(), p, tickers, manual_mode=False)
                        
                        # Sleep to prevent API ban
                        await asyncio.sleep(10)

                # Sleep to avoid double triggering in the same minute
                await asyncio.sleep(60)
                
        except Exception as e:
            print(f"⚠️ Scheduler Error: {e}")
            await asyncio.sleep(60)

@st.cache_resource
def get_bot_app():
    my_persistence = PicklePersistence(filepath='bot_data.pickle', update_interval=1)
    app = ApplicationBuilder().token(TG_TOKEN).persistence(my_persistence).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('stats', stats_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    return app

def run_bot_in_background(app):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if not app.updater or not app.updater.running:
            # Add Scheduler Task
            loop.create_task(auto_scan_scheduler(app))
            app.run_polling(stop_signals=None, close_loop=False)
    except Exception as e:
        print(f"Bot thread error: {e}")

if __name__ == '__main__':
    st.set_page_config(page_title="Vova Bot", page_icon="🤖")
    st.title("💎 Vova Screener Bot (Singleton)")
    
    bot_app = get_bot_app()
    
    if "bot_thread_started" not in st.session_state:
        bot_thread = threading.Thread(target=run_bot_in_background, args=(bot_app,), daemon=True)
        bot_thread.start()
        st.session_state.bot_thread_started = True
        print("✅ Bot polling thread started.")
    
    ny_tz = pytz.timezone('US/Eastern')
    now_ny = datetime.datetime.now(ny_tz)
    st.metric("USA Market Time", now_ny.strftime("%H:%M"))
    st.success("Bot is running in background.")
