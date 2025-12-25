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

from telegram import (
Update, 
ReplyKeyboardMarkup, 
KeyboardButton, 
constants
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
PicklePersistence
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    PicklePersistence,
    Application
)
import telegram.error

# --- КОНФИГУРАЦИЯ ---
nest_asyncio.apply()

# --- CONFIGURATION ---
logging.basicConfig(
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
level=logging.INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 1. ЗАГРУЗКА СЕКРЕТОВ
# 1. LOAD SECRETS (FIXED INDENTATION)
try:
TG_TOKEN = st.secrets["TG_TOKEN"]
ADMIN_ID = int(st.secrets["ADMIN_ID"])
GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
except Exception as e:
st.error(f"❌ Ошибка секретов: {e}")
st.stop()

# 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
last_scan_time = "Never"
    st.error(f"❌ Secret Error: {e}")
    st.stop()

# Индикаторы (Настройки Pine Script)
# 2. GLOBAL SETTINGS
# Indicators (Pine Script Settings)
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14

# ДЕФОЛТНЫЕ ПАРАМЕТРЫ
DEFAULT_PARAMS = {
'risk_usd': 50.0,
'min_rr': 1.25,
'max_atr': 5.0,
'sma': 200,
'tf': 'Daily',
'new_only': True,
    'risk_usd': 50.0,
    'min_rr': 1.25,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
}

# ==========================================
# 3. МАТЕМАТИКА И ЛОГИКА
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
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: return []

def get_financial_info(ticker):
def format_market_cap(val):
    if not val or pd.isna(val): return "N/A"
    if val >= 1e12: return f"{val/1e12:.2f}T"
    if val >= 1e9: return f"{val/1e9:.2f}B"
    if val >= 1e6: return f"{val/1e6:.2f}M"
    return str(val)
    try:
        if val >= 1e12: return f"{val/1e12:.2f}T"
        if val >= 1e9: return f"{val/1e9:.2f}B"
        if val >= 1e6: return f"{val/1e6:.2f}M"
        return str(val)
    except: return "N/A"

def get_extended_info(ticker):
    # Получаем данные для Dashboard (MC, PE, Change%)
try:
t = yf.Ticker(ticker)
i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None
        
        mc = format_market_cap(i.get('marketCap'))
        pe = i.get('trailingPE') or i.get('forwardPE')
    try:
        t = yf.Ticker(ticker)
        try:
            mc = t.fast_info['market_cap']
        except:
            mc = None
            
        try:
            i = t.info
            pe = i.get('trailingPE') or i.get('forwardPE')
        except:
            pe = None
            
        pe_str = f"{pe:.2f}" if pe else "N/A"
        mc_str = format_market_cap(mc)

        # Change % (быстрая оценка через fast_info если доступно, или history)
        # Для скорости берем просто info, если нет - потом посчитаем по df
        return {"mc": mc, "pe": pe_str}
        return {"mc": mc_str, "pe": pe_str}
    except:
        return {"mc": "N/A", "pe": "N/A"}

# --- INDICATORS ---
def calc_sma(s, l): return s.rolling(l).mean()
@@ -188,32 +204,219 @@
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
        prev_st = s_state; prev_cr = s_crit; prev_sh = s_h; prev_sl = s_l
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1:
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                pk_hh = is_hh
                last_pk = prev_sh
                s_state = -1
                s_h = h; s_l = l
                s_crit = h
            else:
                is_hl = True if np.isnan(last_tr) else (prev_sl > last_tr)
                tr_hl = is_hl; last_tr = prev_sl; s_state = 1; s_h = h; s_l = l; s_crit = l
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
    errs = []
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(r['SMA']) or r['Close'] <= r['SMA']: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(r['Peak']) or np.isnan(r['Crit']): errs.append("NO DATA")
    if errs: return False, {}, " ".join(errs)

    # Собираем данные для расчета стопа, даже если сигнал плохой (для дашборда)
price = r['Close']; tp = r['Peak']; crit = r['Crit']; atr = r['ATR']
    r = df.iloc[idx]
    price = r['Close']; tp = r['Peak']; crit = r['Crit']; atr = r['ATR']
    sma = r['SMA']

sl_struct = crit
sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr)
    sl_struct = crit
    sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr) if not np.isnan(sl_struct) else sl_atr

risk = price - final_sl; reward = tp - price
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    risk = price - final_sl; reward = tp - price
    rr = reward / risk if risk > 0 else 0

    rr = reward / risk
    return True, {
    
    data = {
"P": price, "TP": tp, "SL": final_sl, 
"RR": rr, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": rr, "ATR": atr, "Crit": crit,
        "Seq": r['Seq'], "Trend": r['Trend'],
        "SMA": sma, "Struct": r['Struct'],
        "Close": price
    }

    # Валидация
    errs = []
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(sma) or price <= sma: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(tp) or np.isnan(crit): errs.append("NO DATA")
    if risk <= 0: errs.append("BAD STOP")
    if r['Seq'] != 1: errs.append("Seq❌")
    if np.isnan(sma) or price <= sma: errs.append("MA❌")
    if r['Trend'] == -1: errs.append("Trend❌")
    if not r['Struct']: errs.append("Struct❌")

    valid = len(errs) == 0
    return valid, data, errs

# ==========================================
# 4. UI: DASHBOARD STYLE
# 4. CARD FORMATTING
# ==========================================

def format_dashboard_card(ticker, d, shares, is_new, info, p):
    tv_ticker = ticker.replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"

    # 1. EMOJIS & LIGHTS
    # ATR Light
    atr_pct = (d['ATR'] / d['Close']) * 100
    atr_emo = "🟢"
    if atr_pct > 5.0: atr_emo = "🔴"
    elif atr_pct >= 3.0: atr_emo = "🟡"

    # Trend Light
    trend_emo = "🟢" if d['Trend'] == 1 else ("🔴" if d['Trend'] == -1 else "🟡")
    
    # Seq Light
    seq_emo = "🟢" if d['Seq'] == 1 else ("🔴" if d['Seq'] == -1 else "🟡")
    
    # MA Light
    ma_emo = "🟢" if d['Close'] > d['SMA'] else "🔴"

    # 2. VALIDATION
    cond_seq = d['Seq'] == 1
    cond_ma = d['Close'] > d['SMA']
    cond_trend = d['Trend'] != -1
    cond_struct = d['Struct']
    
    is_valid = cond_seq and cond_ma and cond_trend and cond_struct

    # 3. HTML BUILD
    html = f"🖥 <b><a href='{tv_link}'>{ticker}</a></b>\n"
    html += f"PE: {info['pe']} | MC: {info['mc']}\n"
    html += f"ATR: {d['ATR']:.4f} ({atr_pct:.2f}%) {atr_emo}\n"
@@ -196,7 +266,6 @@
        val_pos = shares * d['P']
        profit = (d['TP'] - d['P']) * shares
        loss = (d['P'] - d['SL']) * shares
        
        status = "🆕 NEW" if is_new else "♻️ ACTIVE"

        html += f"{status}\n"
@@ -205,378 +274,209 @@
        html += f"🎯 <b>TP</b>:  <code>{d['TP']:.2f}</code> (<code>+${profit:.0f}</code>)\n"
        html += f"⚖️ <b>RR</b>: <code>{d['RR']:.2f}</code> | Size: <code>{shares}</code>\n"
    else:
        # FAIL REASONS (Как на скрине)
        reasons = []
        if not cond_seq: reasons.append("Seq❌")
        if not cond_ma: reasons.append("MA❌")
        if not cond_trend: reasons.append("Trend❌")
        if not cond_struct: reasons.append("Struct❌")
        
        fail_str = " ".join(reasons) if reasons else "RISK/DATA❌"
        html += f"<b>NO SETUP:</b> {fail_str}"

    return html

# ==========================================
# 5. SCAN PROCESS (AUTO & MANUAL)
# 5. SCANNING PROCESS
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_mode=False):
    chat_id = update.effective_chat.id
    
    start_txt = f"🔎 <b>Scanning {len(tickers)} tickers...</b>"
    status_msg = await context.bot.send_message(chat_id=chat_id, text=start_txt, parse_mode=constants.ParseMode.HTML)
    status_msg = await context.bot.send_message(chat_id=chat_id, text=f"🔎 <b>Scanning {len(tickers)} tickers...</b>", parse_mode='HTML')

    results_found = 0
    total = len(tickers)
    scan_p = p.copy() 

    gc.collect()

    
    for i, t in enumerate(tickers):
        if not context.user_data.get('scanning', False):
            await context.bot.send_message(chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break

        # Update progress bar every 10 tickers
        # UI Update every 10 tickers
        if i % 10 == 0 or i == total - 1:
            try:
                pct = int((i + 1) / total * 10)
                bar = "█" * pct + "░" * (10 - pct)
                await status_msg.edit_text(
                    f"<b>SCAN:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n"
                    f"<i>{t}</i>", 
                    parse_mode='HTML'
                )
                await status_msg.edit_text(f"<b>SCAN:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n<i>{t}</i>", parse_mode='HTML')
            except: pass

        if i % 50 == 0: gc.collect()

        try:
            await asyncio.sleep(0.01) 
            
            inter = "1d" if scan_p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if scan_p['tf'] == "Daily" else "5y"
            # Data Fetch
            await asyncio.sleep(0.01)
            inter = "1d" if p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if p['tf'] == "Daily" else "5y"

            # --- DATA FETCHING ---
            df = yf.download(
                t, 
                period=fetch_period, 
                interval=inter, 
                progress=False, 
                auto_adjust=False, 
                multi_level_index=False
            )
            df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)

            if len(df) < scan_p['sma'] + 5:
                # Если ручной режим - пишем ошибку
                if manual_mode:
                    await context.bot.send_message(chat_id, f"⚠️ {t}: Not enough data", parse_mode='HTML')
            if len(df) < p['sma'] + 5:
                if manual_mode: await context.bot.send_message(chat_id, f"⚠️ <b>{t}</b>: Not enough data", parse_mode='HTML')
                continue

            # --- LOGIC ---
            df = run_vova_logic(df, scan_p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # Analyze Current
            # Logic
            df = run_vova_logic(df, p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            valid, d, errs = analyze_trade(df, -1)
            
            # Check if New
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev

            # --- FILTERING LOGIC ---
            # Auto Mode: Показываем только valid
            # Manual Mode: Показываем ВСЕ (диагностика)
            
            # Decision
            show_card = False
            
            if manual_mode:
                show_card = True # Всегда показываем для диагностики
            else:
                # Auto logic
                if valid:
                    if scan_p['new_only'] and not is_new: 
                        show_card = False
                    else:
                        # Доп фильтры
                        if d['RR'] >= scan_p['min_rr'] and (d['ATR']/d['P'])*100 <= scan_p['max_atr']:
                             risk_per_share = d['P'] - d['SL']
                             if risk_per_share > 0:
                                 shares = int(scan_p['risk_usd'] / risk_per_share)
                                 if shares >= 1: show_card = True
                show_card = True
            elif valid:
                if p['new_only'] and not is_new:
                    show_card = False
                elif d['RR'] >= p['min_rr'] and (d['ATR']/d['P'])*100 <= p['max_atr']:
                    shares = int(p['risk_usd'] / (d['P'] - d['SL']))
                    if shares >= 1: show_card = True

            if show_card:
                info = get_extended_info(t)
                risk_per_share = d['P'] - d['SL']
                shares = 0
                if risk_per_share > 0:
                    shares = int(scan_p['risk_usd'] / risk_per_share)
                
                card = format_dashboard_card(t, d, shares, is_new, info, scan_p['risk_usd'])
                await context.bot.send_message(chat_id=chat_id, text=card, parse_mode=constants.ParseMode.HTML, disable_web_page_preview=True)
                shares = int(p['risk_usd'] / risk_per_share) if risk_per_share > 0 else 0
                card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'])
                await context.bot.send_message(chat_id=chat_id, text=card, parse_mode='HTML', disable_web_page_preview=True)
                results_found += 1

        except Exception as e:
            if manual_mode:
                # await context.bot.send_message(chat_id, f"⚠️ {t}: Error {e}")
                pass
            if manual_mode: await context.bot.send_message(chat_id, f"⚠️ <b>{t} Error:</b> {str(e)}", parse_mode='HTML')
            continue

    global last_scan_time
    last_scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    final_txt = (
        f"🏁 <b>SCAN COMPLETE</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"✅ <b>Found:</b> {results_found}\n"
        f"📊 <b>Total:</b> {total}\n"
    )
    await context.bot.send_message(chat_id=chat_id, text=final_txt, parse_mode='HTML')
    await context.bot.send_message(chat_id=chat_id, text=f"🏁 <b>SCAN COMPLETE</b>\n✅ Found: {results_found}", parse_mode='HTML')
    context.user_data['scanning'] = False

# ==========================================
# 4. HELPER FUNCTIONS & UI
# 6. HELPER FUNCTIONS & HANDLERS
# 6. BOT HANDLERS & HELPERS
# ==========================================

def is_market_open():
@@ -241,9 +444,6 @@
context.bot_data['active_users'].add(user_id)
allowed = get_allowed_users()
if user_id not in allowed:
        msg = f"⛔ <b>Access Denied</b>\n\nID: <code>{user_id}</code>\nSend ID to: <b>@Vova_Skl</b>"
        try: await update.message.reply_html(msg)
        except: pass
return False
return True

@@ -257,34 +457,6 @@
context.user_data['params'] = new_params
return context.user_data['params']

# --- UI ---
def format_luxury_card(ticker, d, shares, is_new, pe_val, risk_usd):
    tv_ticker = ticker.replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
    status = "🆕 NEW" if is_new else "♻️ ACTIVE"
    pe_str = f"{pe_val:.1f}" if pe_val else "N/A"
    
    val_pos = shares * d['P']
    profit = (d['TP'] - d['P']) * shares
    loss = (d['P'] - d['SL']) * shares
    atr_pct = (d['ATR'] / d['P']) * 100
    
    html = (
        f"💎 <b><a href='{tv_link}'>{ticker}</a></b> | {status}\n"
        f"💵 <b>${d['P']:.2f}</b> (P/E: {pe_str})\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>POSITION</b>\n"
        f"• Shares: <code>{shares}</code>\n"
        f"• Value:  <code>${val_pos:.0f}</code>\n"
        f"• R:R:    <code>{d['RR']:.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TP</b>:  <code>{d['TP']:.2f}</code> (<code>+${profit:.0f}</code>)\n"
        f"🛑 <b>SL</b>:  <code>{d['SL']:.2f}</code> (<code>-${abs(loss):.0f}</code>)\n"
        f"📉 <b>Critical Level</b>: <code>{d['Crit']:.2f}</code>\n"
        f"⚡ <b>ATR Vol</b>: <code>{d['ATR']:.2f}</code> (<code>{atr_pct:.1f}%</code>)"
    )
    return html
def get_allowed_users():
    allowed = {ADMIN_ID}
    if not GITHUB_USERS_URL: return allowed
    try:
        response = requests.get(GITHUB_USERS_URL, timeout=5)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.strip().isdigit(): allowed.add(int(line.strip()))
    except: pass
    return allowed

async def check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in get_allowed_users(): return False
    return True

async def safe_get_params(context):
    if 'params' not in context.user_data: context.user_data['params'] = DEFAULT_PARAMS.copy()
    return context.user_data['params']

def get_reply_keyboard(p):
risk_txt = f"💸 Risk: ${p['risk_usd']:.0f}"
rr_txt = f"⚖️ RR: {p['min_rr']}"
@@ -323,112 +495,9 @@
"<b>📊 ATR %</b>: Max volatility allowed.\n"
"<b>📈 SMA</b>: Trend filter (Price > SMA).\n"
"<b>✨ Only New</b>: \n✅ = Show only fresh signals from TODAY.\n❌ = Show ALL valid signals found.\n"
        "<b>🔍 Manual Scan</b>: Send tickers comma separated (e.g. <code>AAPL, TSLA</code>) to see DIAGNOSIS."
)
    risk = f"💸 Risk: ${p['risk_usd']:.0f}"
    rr = f"⚖️ RR: {p['min_rr']}"
    atr = f"📊 ATR: {p['max_atr']}%"
    sma = f"📈 SMA: {p['sma']}"
    tf = "📅 Daily" if p['tf'] == 'Daily' else "🗓 Weekly"
    new = f"Only New {'✅' if p['new_only'] else '❌'}"
    return ReplyKeyboardMarkup([
        [KeyboardButton(risk), KeyboardButton(rr)],
        [KeyboardButton(atr), KeyboardButton(sma)],
        [KeyboardButton(tf), KeyboardButton(new)], 
        [KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP SCAN")],
        [KeyboardButton("ℹ️ HELP / INFO")]
    ], resize_keyboard=True)

# ==========================================
# 5. SCAN PROCESS (MANUAL ONLY)
# ==========================================
async def run_scan_process(update, context, p, tickers):
    start_txt = "🚀 <b>Scanning Started...</b>"
    chat_id = update.effective_chat.id
    
    status_msg = await context.bot.send_message(chat_id=chat_id, text=start_txt, parse_mode=constants.ParseMode.HTML)
    
    results_found = 0
    total = len(tickers)
    scan_p = p.copy() 

    gc.collect()

    for i, t in enumerate(tickers):
        if not context.user_data.get('scanning', False):
            await context.bot.send_message(chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break

        if i % 10 == 0 or i == total - 1:
            pct = int((i + 1) / total * 10)
            bar = "█" * pct + "░" * (10 - pct)
            try:
                await status_msg.edit_text(
                    f"<b>SCAN:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n"
                    f"<i>SMA{scan_p['sma']} | {scan_p['tf']}</i>", 
                    parse_mode='HTML'
                )
            except: pass
            
        if i % 50 == 0: gc.collect()

        try:
            await asyncio.sleep(0.01) 
            
            inter = "1d" if scan_p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if scan_p['tf'] == "Daily" else "5y"
            
            # --- DATA FETCHING ---
            df = yf.download(
                t, 
                period=fetch_period, 
                interval=inter, 
                progress=False, 
                auto_adjust=False, 
                multi_level_index=False
            )
            
            if len(df) < scan_p['sma'] + 5:
                continue

            # --- LOGIC ---
            df = run_vova_logic(df, scan_p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # 1. Analyze Current Candle
            valid, d, reason = analyze_trade(df, -1)
            
            if not valid:
                continue

            # 2. Check if New
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if scan_p['new_only'] and not is_new: continue
            
            # 3. Parameters
            if d['RR'] < scan_p['min_rr']: continue
            if (d['ATR']/d['P'])*100 > scan_p['max_atr']: continue
            
            # 4. Risk
            risk_per_share = d['P'] - d['SL']
            if risk_per_share <= 0: continue
            shares = int(scan_p['risk_usd'] / risk_per_share)
            if shares < 1: 
                continue
            
            # --- FOUND ---
            pe = get_financial_info(t)
            card = format_luxury_card(t, d, shares, is_new, pe, scan_p['risk_usd'])
            
            await context.bot.send_message(chat_id=chat_id, text=card, parse_mode=constants.ParseMode.HTML, disable_web_page_preview=True)
            results_found += 1
            
        except Exception:
            pass

    global last_scan_time
    last_scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    final_txt = (
        f"🏁 <b>SCAN COMPLETE</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"✅ <b>Found:</b> {results_found} signals\n"
        f"📊 <b>Total Scanned:</b> {total}\n"
    )
    await context.bot.send_message(chat_id=chat_id, text=final_txt, parse_mode='HTML')
    context.user_data['scanning'] = False
    await context.bot.send_message(chat_id=chat_id, text=get_status_text("Ready", p), reply_markup=get_reply_keyboard(p), parse_mode='HTML')

# ==========================================
# 6. HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not await check_auth(update, context): return
p = await safe_get_params(context)
@@ -438,16 +507,14 @@
welcome_txt = (
f"👋 <b>Welcome, {update.effective_user.first_name}!</b>\n\n"
f"💎 <b>Vova Screener Bot</b> is ready.\n"
        f"Use the menu below to configure parameters and start scanning.\n\n"
        f"<i>Tap 'Start Scan' to begin.</i>"
        f"Tap 'Start Scan' to begin or send tickers manually."
)
await update.message.reply_html(welcome_txt, reply_markup=get_reply_keyboard(p))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
if update.effective_user.id != ADMIN_ID: return
active = context.bot_data.get('active_users', set())
    allowed = get_allowed_users()
    msg = f"📊 <b>ADMIN STATS</b>\nActive: {len(active)}\nWhitelist: {len(allowed)}\nLast Scan: {last_scan_time}"
    msg = f"📊 <b>ADMIN STATS</b>\nActive: {len(active)}\nLast Scan: {last_scan_time}"
await update.message.reply_html(msg)
    if not await check_auth(update, context): return
    p = await safe_get_params(context)
    await update.message.reply_html(f"👋 Welcome! Bot is ready.", reply_markup=get_reply_keyboard(p))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
@@ -462,7 +529,7 @@
return
context.user_data['scanning'] = True
tickers = get_sp500_tickers()
    if not await check_auth(update, context): return
    text = update.message.text
    p = await safe_get_params(context)
    
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'): return await update.message.reply_text("⚠️ Already running!")
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers))
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=False))
return

elif text == "⏹ STOP SCAN":
@@ -512,48 +579,51 @@
await update.message.reply_text("❌ Invalid number. Try again.")
return

    # MANUAL TICKER SCAN
elif "," in text or (text.isalpha() and len(text) < 6):
ts = [x.strip().upper() for x in text.split(",") if x.strip()]
if ts:
            await update.message.reply_text(f"🔎 Scanning: {ts}")
            await run_scan_process(update, context, p, ts)
        return
    elif text == "⏹ STOP SCAN":
        context.user_data['scanning'] = False
        return await update.message.reply_text("🛑 Stopping...")
    elif text == "ℹ️ HELP / INFO":
        return await update.message.reply_html("📚 <b>Commands:</b>\nSend tickers (e.g. <code>AAPL, TSLA</code>) for diagnosis.")
    elif "Daily" in text or "Weekly" in text: p['tf'] = "Weekly" if p['tf'] == "Daily" else "Daily"
    elif "Only New" in text: p['new_only'] = not p['new_only']
    elif "SMA:" in text:
        opts = [100, 150, 200]
        try: p['sma'] = opts[(opts.index(int(text.split(":")[1].strip())) + 1) % 3]
        except: p['sma'] = 200
    elif "Risk:" in text:
        context.user_data['input_mode'] = "risk"
        return await update.message.reply_text("✏️ Enter Risk $:")
    elif context.user_data.get('input_mode') == "risk":
        try: p['risk_usd'] = max(1.0, float(text)); context.user_data['input_mode'] = None; await update.message.reply_text("✅ Updated")
        except: pass
    elif "," in text or (text.isalpha() and len(text) < 6):
        ts = [x.strip().upper() for x in text.split(",") if x.strip()]
        if ts:
            context.user_data['scanning'] = True
            await context.bot.send_message(update.effective_chat.id, f"🔎 Diagnosing: {ts}")
            # manual_mode=True включает показ ВСЕХ карточек (даже с ошибками)
            await run_scan_process(update, context, p, ts, manual_mode=True)
return
        return

context.user_data['params'] = p
await update.message.reply_text(get_status_text("Ready", p), reply_markup=get_reply_keyboard(p), parse_mode='HTML')
    context.user_data['params'] = p
    await update.message.reply_text(f"Config: Risk ${p['risk_usd']} | {p['tf']}", reply_markup=get_reply_keyboard(p))

# ==========================================
# 7. MAIN (ИСПРАВЛЕНО - ЗАПУСК В ФОНЕ)
# 7. MAIN (BACKGROUND RUN FIX)
# 7. ARCHITECTURE: SINGLETON BOT (Fix for Conflict)
# ==========================================
if __name__ == '__main__':
st.set_page_config(page_title="Vova Bot", page_icon="🤖")
st.title("💎 Vova Screener Bot")

ny_tz = pytz.timezone('US/Eastern')
now_ny = datetime.datetime.now(ny_tz)
market_open = is_market_open()
c1, c2 = st.columns(2)
with c1: st.metric("USA Market", "OPEN" if market_open else "CLOSED", delta=now_ny.strftime("%H:%M NY"))
with c2: st.metric("Bot Status", "Running")

if "bot_active" not in st.session_state:
st.session_state.bot_active = True

my_persistence = PicklePersistence(filepath='bot_data.pickle', update_interval=1)
application = ApplicationBuilder().token(TG_TOKEN).persistence(my_persistence).build()

application.add_handler(CommandHandler('start', start))
application.add_handler(CommandHandler('stats', stats_command))
application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
# Use st.cache_resource to enforce only ONE bot instance per Streamlit container
@st.cache_resource
def get_bot_app():
    """Create the Application ONCE and return it."""
    my_persistence = PicklePersistence(filepath='bot_data.pickle', update_interval=1)
    app = ApplicationBuilder().token(TG_TOKEN).persistence(my_persistence).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    return app

def run_bot_in_background(app):
    """Run polling in a separate daemon thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Check if already running to double-prevent conflicts
        if not app.updater or not app.updater.running:
            app.run_polling(stop_signals=None, close_loop=False)
    except Exception as e:
        print(f"Bot thread error: {e}")

try:
loop = asyncio.get_event_loop()
except RuntimeError:
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

loop.create_task(application.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False, stop_signals=None))

print("✅ Бот запущен в фоне")
else:
st.info("Бот активен и работает в фоне.")
if __name__ == '__main__':
    st.set_page_config(page_title="Vova Bot", page_icon="🤖")
    st.title("💎 Vova Screener Bot (Singleton)")
    
    # 1. Get the Singleton Bot Instance
    bot_app = get_bot_app()
    
    # 2. Start Polling in a Thread (Non-blocking)
    if "bot_thread_started" not in st.session_state:
        # Create a thread that runs the asyncio loop for the bot
        bot_thread = threading.Thread(target=run_bot_in_background, args=(bot_app,), daemon=True)
        bot_thread.start()
        st.session_state.bot_thread_started = True
        print("✅ Bot polling thread started.")
    
    # 3. UI Display
    ny_tz = pytz.timezone('US/Eastern')
    now_ny = datetime.datetime.now(ny_tz)
    st.metric("USA Market Time", now_ny.strftime("%H:%M"))
    st.success("Bot is running in background.")
