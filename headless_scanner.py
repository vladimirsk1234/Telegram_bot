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

from numba import jit, float64, int64, boolean
import numpy as np

# ==========================================
# NUMBA ENGINE (Скорость x500)
# ==========================================
@jit(nopython=True, cache=True)
def calculate_structure_engine(c_a, h_a, l_a):
    n = len(c_a)
    
    # Инициализация массивов для результатов
    seq_st = np.zeros(n, dtype=np.int64)
    crit_lvl = np.full(n, np.nan, dtype=np.float64)
    res_peak = np.full(n, np.nan, dtype=np.float64)
    res_struct = np.zeros(n, dtype=np.bool_) 
    
    # Переменные состояния
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]
    s_l = l_a[0]
    
    last_pk = np.nan
    last_tr = np.nan
    
    pk_hh = False
    tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h 
        prev_sl = s_l 
        
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr):
            if c < prev_cr: brk = True
        elif prev_st == -1 and not np.isnan(prev_cr):
            if c > prev_cr: brk = True
            
        if brk:
            if prev_st == 1: 
                final_high = max(prev_sh, h)
                if np.isnan(last_pk): is_hh = True
                else: is_hh = (final_high > last_pk)
                pk_hh = is_hh
                last_pk = final_high 
                s_state = -1
                s_h = h; s_l = l
                s_crit = h
            else: 
                final_low = min(prev_sl, l)
                if np.isnan(last_tr): is_hl = True
                else: is_hl = (final_low > last_tr)
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
        if pk_hh and tr_hl: res_struct[i] = True
        else: res_struct[i] = False
            
    return seq_st, crit_lvl, res_peak, res_struct

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
# 1. LOAD SECRETS
# ==========================================
try:
    TG_TOKEN = st.secrets["TG_TOKEN"].strip()
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "").strip()
    CHANNEL_ID = st.secrets.get("CHANNEL_ID", None)
    if CHANNEL_ID: CHANNEL_ID = int(CHANNEL_ID)
    print(f"✅ Loaded Token: {TG_TOKEN[:5]}... | Admin ID: {ADMIN_ID} | Channel: {CHANNEL_ID}")
except Exception as e:
    st.error(f"❌ Secret Error: {e}")
    st.stop()

# 2. GLOBAL SETTINGS
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14
SMA_MAJ = 200

DEFAULT_PARAMS = {
    'risk_usd': 100.0,
    'min_rr': 1.5,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
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
    
    # 2. Подготовка данных для Numba
    c_vals = df['Close'].values.astype(np.float64)
    h_vals = df['High'].values.astype(np.float64)
    l_vals = df['Low'].values.astype(np.float64)
    
    # 3. ЗАПУСК БЫСТРОГО ДВИЖКА (NUMBA)
    seq_st, crit_lvl, res_peak, res_struct = calculate_structure_engine(c_vals, h_vals, l_vals)
    
    # 4. Сборка логики тренда
    adx_str = adx >= adx_thr
    
    ema_f_prev = ema_f.shift(1)
    ema_s_prev = ema_s.shift(1)
    hist_prev = hist.shift(1)
    
    bull_mom = (ema_f > ema_f_prev) & (ema_s > ema_s_prev) & (hist > hist_prev) & (efi > 0)
    bear_mom = (ema_f < ema_f_prev) & (ema_s < ema_s_prev) & (hist < hist_prev) & (efi < 0)
    
    bull = (adx_str & (p_di > m_di)) & bull_mom
    bear = (adx_str & (m_di > p_di)) & bear_mom
    
    t_st = np.zeros(len(df), dtype=int)
    t_st[bull] = 1
    t_st[bear] = -1
    
    df['Seq'] = seq_st
    df['Crit'] = crit_lvl
    df['Peak'] = res_peak
    df['Struct'] = res_struct
    df['Trend'] = t_st
    df['ATR'] = atr
    
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
# 4. UI: DASHBOARD STYLE (UPDATED)
# ==========================================
def format_dashboard_card(ticker, d, shares, is_new, info, p_risk, sma_len, public_view=False):
    tv_ticker = ticker.replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
    
    # ---------------------------------------------------------
    # ВАРИАНТ 1: ПУБЛИЧНЫЙ КАНАЛ (SIMPLE SIGNAL)
    # ---------------------------------------------------------
    if public_view:
        setup_type = "LONG 📈" if d['Trend'] == 1 else "SHORT 📉"
        
        html = (
            f"<b>{setup_type}: <a href='{tv_link}'>{ticker}</a></b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"🛒 <b>Entry:</b> {d['P']:.2f}\n\n"
            f"🎯 <b>TP:</b> {d['TP']:.2f}\n"
            f"🛑 <b>SL:</b> {d['SL']:.2f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚖️ <b>R/R:</b> {d['RR']:.2f}R"
        )
        return html

    # ---------------------------------------------------------
    # ВАРИАНТ 2: ЛИЧНЫЙ БОТ (FULL TRADER DASHBOARD)
    # ---------------------------------------------------------
    pe_str = str(info.get('pe', 'N/A'))
    mc_str = str(info.get('mc', 'N/A'))
    atr_pct = (d['ATR'] / d['Close']) * 100
    
    trend_emo = "🟢" if d['Trend'] == 1 else ("🔴" if d['Trend'] == -1 else "🟡")
    seq_emo = "🟢" if d['Seq'] == 1 else ("🔴" if d['Seq'] == -1 else "🟡")
    ma_emo = "🟢" if d['Close'] > d['SMA'] else "🔴"
    status_icon = "🆕" if is_new else "♻️"

    # Расчет PnL в долларах
    risk_val = d['P'] - d['SL']
    reward_val = d['TP'] - d['P']
    profit = reward_val * shares
    loss = risk_val * shares
    total_val = shares * d['P']

    header = f"<b><a href='{tv_link}'>{ticker}</a></b>  ${d['P']:.2f}\n"
    
    context_block = (
        f"MC: {mc_str} | P/E: {pe_str}\n"
        f"ATR: ${d['ATR']:.2f} ({atr_pct:.2f}%)\n"
        f"Trend {trend_emo}  Seq {seq_emo}  MA{sma_len} {ma_emo}\n"
    )

    size_line = f"Size: <b>{shares}</b> shares (${total_val:,.0f})\n"
    sl_line = f"🛑 SL: {d['SL']:.2f}  (-${loss:.0f})\n"
    tp_line = f"🎯 TP: {d['TP']:.2f}  (+${profit:.0f})\n"

    html = (
        f"{status_icon} {header}"
        f"{size_line}"
        f"{context_block}"
        f"{sl_line}"
        f"{tp_line}"
        f"⚖️ RR: {d['RR']:.2f}"
    )
    
    return html

# ==========================================
# 5. SCANNING PROCESS
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_mode=False, is_auto=False):
    private_chat_id = update.effective_chat.id if update.effective_chat else ADMIN_ID
    
    ny_tz = pytz.timezone('US/Eastern')
    today_str = datetime.datetime.now(ny_tz).strftime('%Y-%m-%d')
    
    if 'channel_mem' not in context.bot_data: 
        context.bot_data['channel_mem'] = {'date': today_str, 'tickers': []}
    if context.bot_data['channel_mem']['date'] != today_str:
        context.bot_data['channel_mem'] = {'date': today_str, 'tickers': []}

    if not is_auto:
        config_display = (
            f"⚙️ <b>Active Settings:</b>\n"
            f"Risk: ${p['risk_usd']:.0f} | RR: {p['min_rr']}\n"
            f"SMA: {p['sma']} | ATR Max: {p['max_atr']}%\n"
            f"TF: {p['tf']} | New Only: {'✅' if p['new_only'] else '❌'}"
        )
        status_msg = await context.bot.send_message(
            chat_id=private_chat_id, 
            text=f"🚀 <b>Scan Started...</b>\n\n{config_display}", 
            parse_mode='HTML'
        )
    
    results_found = 0
    total = len(tickers)
    
    for i, t in enumerate(tickers):
        # Stop Logic (Manual Only)
        if not is_auto and not context.user_data.get('scanning', False):
            await context.bot.send_message(private_chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break
            
        # UI Update (Manual Only)
        if not is_auto and (i % 10 == 0 or i == total - 1):
            try:
                pct = int((i + 1) / total * 10)
                bar = "█" * pct + "░" * (10 - pct)
                percent_num = int((i + 1) / total * 100)
                await status_msg.edit_text(
                    f"🔎 <b>Scanning S&P 500...</b>\n"
                    f"[{bar}] {percent_num}%\n"
                    f"👉 Checking: <b>{t}</b> ({i+1}/{total})\n\n"
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
            
            if len(df) < p['sma'] + 5: continue
            
            df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            valid, d, errs = analyze_trade(df, -1)
            
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            show_card = False
            
            if manual_mode: 
                show_card = True 
            elif valid:
                passes_filters = (d['RR'] >= p['min_rr'] and (d['ATR']/d['P'])*100 <= p['max_atr'])
                if passes_filters:
                    if p['new_only'] and not is_new: show_card = False
                    else: show_card = True

            if show_card:
                info = get_extended_info(t)
                await asyncio.sleep(0.5)
                risk_per_share = d['P'] - d['SL']
                shares = int(p['risk_usd'] / risk_per_share) if risk_per_share > 0 else 0
                
                # 1. AUTO SCAN -> CHANNEL (PUBLIC VIEW)
                if is_auto and CHANNEL_ID:
                    if t not in context.bot_data['channel_mem']['tickers']:
                        # --- PUBLIC VIEW IS TRUE HERE ---
                        public_card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'], p['sma'], public_view=True)
                        
                        final_msg = public_card + "\n\n💎 <i>Join Premium for Live Alerts!</i>"
                        await context.bot.send_message(chat_id=CHANNEL_ID, text=final_msg, parse_mode='HTML', disable_web_page_preview=True)
                        context.bot_data['channel_mem']['tickers'].append(t)
                        results_found += 1
                        
                # 2. MANUAL SCAN -> PRIVATE USER (FULL VIEW)
                elif not is_auto:
                      card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'], p['sma'], public_view=False)
                      await context.bot.send_message(chat_id=private_chat_id, text=card, parse_mode='HTML', disable_web_page_preview=True)
                      results_found += 1
                
        except Exception as e:
            if manual_mode: await context.bot.send_message(private_chat_id, f"⚠️ {t}: {e}")
            continue
    
    if not is_auto:
        context.user_data['scanning'] = False
        await context.bot.send_message(private_chat_id, f"✅ <b>Scan Complete.</b>\nFound: {results_found} signals.", parse_mode='HTML')

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
            msg = (f"🛑 <b>Authorization Required</b>\n\n🆔 <b>Your ID:</b> <code>{user_id}</code>")
            await update.message.reply_html(msg)
        except: pass
        return False
    
    if 'active_users' not in context.bot_data: context.bot_data['active_users'] = set()
    context.bot_data['active_users'].add(user_id)
    return True

async def safe_get_params(context):
    if 'params' not in context.user_data: context.user_data['params'] = DEFAULT_PARAMS.copy()
    p = context.user_data['params']
    return p

# --- KEYBOARDS ---
def get_main_keyboard(p):
    risk = f"💸 Risk: ${p['risk_usd']:.0f}"
    rr = f"⚖️ RR: {p['min_rr']}"
    atr = f"📊 ATR Max: {p['max_atr']}%"
    sma = f"📈 SMA: {p['sma']}"
    tf = f"⏳ TIMEFRAME: {p['tf'][0]}"
    new = f"Only New {'✅' if p['new_only'] else '❌'}"
    
    return ReplyKeyboardMarkup([
        [KeyboardButton(risk), KeyboardButton(rr)],
        [KeyboardButton(atr), KeyboardButton(sma)],
        [KeyboardButton(tf), KeyboardButton(new)], 
        [KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP SCAN")],
        [KeyboardButton("ℹ️ HELP / INFO")]
    ], resize_keyboard=True)

def get_sma_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton("SMA 100"), KeyboardButton("SMA 150"), KeyboardButton("SMA 200")], [KeyboardButton("🔙 Back")]], resize_keyboard=True)

def get_tf_keyboard():
    return ReplyKeyboardMarkup([[KeyboardButton("Daily (D)"), KeyboardButton("Weekly (W)")], [KeyboardButton("🔙 Back")]], resize_keyboard=True)

# --- COMMANDS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    try: p = await safe_get_params(context)
    except: p = DEFAULT_PARAMS.copy(); context.user_data['params'] = p

    context.user_data['input_mode'] = None
    welcome_text = f"👋 <b>Welcome!</b>\n\nStrategy: Sequence + Momentum + Volatility.\nConfigure below."
    await update.message.reply_html(welcome_text, reply_markup=get_main_keyboard(p))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    allowed = get_allowed_users()
    active = context.bot_data.get('active_users', set())
    msg = (f"📊 <b>BOT STATISTICS</b>\n✅ Approved: {len(allowed)}\n👥 Active: {len(active)}")
    await update.message.reply_html(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    text = update.message.text
    p = await safe_get_params(context)
    
    if text == "🔙 Back":
        context.user_data['input_mode'] = None
        await update.message.reply_text("🔙 Main Menu", reply_markup=get_main_keyboard(p))
        return

    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'): return await update.message.reply_text("⚠️ Already running!")
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=False, is_auto=False)) 
        return

    elif text == "⏹ STOP SCAN":
        context.user_data['scanning'] = False
        p['auto_scan'] = False 
        context.user_data['params'] = p
        return await update.message.reply_text("🛑 Stopping all scans...", reply_markup=get_main_keyboard(p))

    elif text == "ℹ️ HELP / INFO":
        return await update.message.reply_html("<b>S&P500 SCREENER</b>\nUse buttons to configure risk and filters.")

    elif "SMA:" in text:
        context.user_data['input_mode'] = "sma_select"
        await update.message.reply_text("Select SMA Length:", reply_markup=get_sma_keyboard())
        return
    elif "TIMEFRAME:" in text:
        context.user_data['input_mode'] = "tf_select"
        await update.message.reply_text("Select Timeframe:", reply_markup=get_tf_keyboard())
        return

    if context.user_data.get('input_mode') == "sma_select":
        if text in ["SMA 100", "SMA 150", "SMA 200"]:
            p['sma'] = int(text.split()[1])
            context.user_data['input_mode'] = None
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            await update.message.reply_text(f"✅ SMA set to {p['sma']}", reply_markup=get_main_keyboard(p))
        return
    if context.user_data.get('input_mode') == "tf_select":
        if "Daily" in text: p['tf'] = "Daily"
        elif "Weekly" in text: p['tf'] = "Weekly"
        context.user_data['input_mode'] = None
        context.user_data['params'] = p
        await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
        await update.message.reply_text(f"✅ Timeframe set to {p['tf']}", reply_markup=get_main_keyboard(p))
        return

    if "Only New" in text: 
        p['new_only'] = not p['new_only']
        status = "ENABLED" if p['new_only'] else "DISABLED"
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
            context.user_data['params'] = p
            await context.application.persistence.update_user_data(update.effective_user.id, context.user_data)
            await update.message.reply_text(f"✅ Max ATR updated to {val}%", reply_markup=get_main_keyboard(p))
        except: await update.message.reply_text("❌ Invalid number.")
        return

    if "," in text or (text.isalpha() and len(text) < 6):
        ts = [x.strip().upper() for x in text.split(",") if x.strip()]
        if ts:
            context.user_data['scanning'] = True
            await context.bot.send_message(update.effective_chat.id, f"🔎 Diagnosing: {ts}")
            await run_scan_process(update, context, p, ts, manual_mode=True, is_auto=False)
        return

    context.user_data['params'] = p
    await update.message.reply_text(f"Config: Risk ${p['risk_usd']} | {p['tf']}", reply_markup=get_main_keyboard(p))

# ==========================================
# 7. ARCHITECTURE: SINGLETON BOT + SCHEDULER
# ==========================================

# --- NEW: REUSABLE CHANNEL SCAN LOGIC ---
async def trigger_channel_scan(app):
    print("🚀 Triggering Channel Scan...")
    
    # 🔔 1. УВЕДОМЛЕНИЕ ВАМ: СТАРТ
    try:
        await app.bot.send_message(
            chat_id=ADMIN_ID, 
            text="🔄 <b>Auto-Scan Started...</b>\n<i>Scanning market for Channel signals.</i>",
            parse_mode='HTML'
        )
    except Exception as e: print(f"Notify Error: {e}")

    # Настройки для канала
    channel_params = {
        'risk_usd': 100.0, 'min_rr': 1.5, 'max_atr': 5.0, 'sma': 200,           
        'tf': 'Daily', 'new_only': True, 'auto_scan': True
    }
    
    tickers = get_sp500_tickers()
    
    # Создаем фейковый контекст
    class DummyObj: pass
    u_upd = DummyObj(); u_upd.effective_chat = None 
    u_ctx = DummyObj(); u_ctx.bot = app.bot; u_ctx.user_data = {}; u_ctx.bot_data = app.bot_data 
    
    # Запуск сканирования
    await run_scan_process(u_upd, u_ctx, channel_params, tickers, manual_mode=False, is_auto=True)

    # 🔔 2. УВЕДОМЛЕНИЕ ВАМ: ФИНИШ
    try:
        await app.bot.send_message(
            chat_id=ADMIN_ID, 
            text="✅ <b>Auto-Scan Complete.</b>\n<i>Check the channel for new posts.</i>",
            parse_mode='HTML'
        )
    except: pass

# --- NEW COMMAND: /auto ---
async def force_auto_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    await update.message.reply_text("🚀 <b>Forcing Channel Scan...</b>", parse_mode='HTML')
    await trigger_channel_scan(context.application)

# --- UPDATED SCHEDULER ---
async def auto_scan_scheduler(app):
    print("⏳ Scheduler started... (Target: 10:00 ET)")
    while True:
        try:
            ny_tz = pytz.timezone('US/Eastern')
            now = datetime.datetime.now(ny_tz)
            is_market_day = now.weekday() < 5 
            is_scan_time = (now.hour == 10 and now.minute == 0)
            
            if is_market_day and is_scan_time:
                print("⏰ Time match! Auto-Scan Triggered.")
                await trigger_channel_scan(app)
                await asyncio.sleep(61)
            
            await asyncio.sleep(30)
        except Exception as e:
            print(f"Scheduler Error: {e}")
            await asyncio.sleep(60)
            
@st.cache_resource
def get_bot_app():
    my_persistence = PicklePersistence(filepath='bot_data.pickle', update_interval=1)
    app = ApplicationBuilder().token(TG_TOKEN).persistence(my_persistence).build()
    
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('stats', stats_command))
    # ✅ REGISTER NEW COMMAND
    app.add_handler(CommandHandler('auto', force_auto_scan))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    return app

def run_bot_in_background(app):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if not app.updater or not app.updater.running:
            loop.create_task(auto_scan_scheduler(app))
            app.run_polling(stop_signals=None, close_loop=False)
    except Exception as e:
        print(f"Bot thread error: {e}")

if __name__ == '__main__':
    st.set_page_config(page_title="S&P500 Bot Screener", page_icon="🤖")
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
