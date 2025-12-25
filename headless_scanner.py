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
    PicklePersistence
)
import telegram.error

# --- КОНФИГУРАЦИЯ ---
nest_asyncio.apply()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 1. ЗАГРУЗКА СЕКРЕТОВ
try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
except Exception as e:
    st.error(f"❌ Ошибка секретов: {e}")
    st.stop()

# 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
last_scan_time = "Never"

# Индикаторы (Настройки Pine Script)
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14

# ДЕФОЛТНЫЕ ПАРАМЕТРЫ
DEFAULT_PARAMS = {
    'risk_usd': 50.0,
    'min_rr': 1.25,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
}

# ==========================================
# 3. МАТЕМАТИКА И ЛОГИКА
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
    if val >= 1e12: return f"{val/1e12:.2f}T"
    if val >= 1e9: return f"{val/1e9:.2f}B"
    if val >= 1e6: return f"{val/1e6:.2f}M"
    return str(val)

def get_extended_info(ticker):
    # Получаем данные для Dashboard (MC, PE, Change%)
    try:
        t = yf.Ticker(ticker)
        i = t.info
        
        mc = format_market_cap(i.get('marketCap'))
        pe = i.get('trailingPE') or i.get('forwardPE')
        pe_str = f"{pe:.2f}" if pe else "N/A"
        
        # Change % (быстрая оценка через fast_info если доступно, или history)
        # Для скорости берем просто info, если нет - потом посчитаем по df
        return {"mc": mc, "pe": pe_str}
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

# --- STRATEGY CORE ---
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
    
    # Собираем данные для расчета стопа, даже если сигнал плохой (для дашборда)
    price = r['Close']; tp = r['Peak']; crit = r['Crit']; atr = r['ATR']
    sma = r['SMA']
    
    sl_struct = crit
    sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr) if not np.isnan(sl_struct) else sl_atr
    
    risk = price - final_sl; reward = tp - price
    rr = reward / risk if risk > 0 else 0
    
    data = {
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
    
    valid = len(errs) == 0
    return valid, data, errs

# ==========================================
# 4. UI: DASHBOARD STYLE
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
    html += f"Trend {trend_emo}  Seq {seq_emo}  MA 200 {ma_emo}\n"
    html += "━━━━━━━━━━━━━━━━━━\n"
    
    if is_valid:
        val_pos = shares * d['P']
        profit = (d['TP'] - d['P']) * shares
        loss = (d['P'] - d['SL']) * shares
        
        status = "🆕 NEW" if is_new else "♻️ ACTIVE"
        
        html += f"{status}\n"
        html += f"💵 <b>${d['P']:.2f}</b>\n"
        html += f"🛑 <b>SL</b>:  <code>{d['SL']:.2f}</code> (<code>-${abs(loss):.0f}</code>)\n"
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
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_mode=False):
    chat_id = update.effective_chat.id
    
    start_txt = f"🔎 <b>Scanning {len(tickers)} tickers...</b>"
    status_msg = await context.bot.send_message(chat_id=chat_id, text=start_txt, parse_mode=constants.ParseMode.HTML)
    
    results_found = 0
    total = len(tickers)
    scan_p = p.copy() 

    gc.collect()

    for i, t in enumerate(tickers):
        if not context.user_data.get('scanning', False):
            await context.bot.send_message(chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break
            
        # Update progress bar every 10 tickers
        if i % 10 == 0 or i == total - 1:
            try:
                pct = int((i + 1) / total * 10)
                bar = "█" * pct + "░" * (10 - pct)
                await status_msg.edit_text(
                    f"<b>SCAN:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n"
                    f"<i>{t}</i>", 
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
                # Если ручной режим - пишем ошибку
                if manual_mode:
                    await context.bot.send_message(chat_id, f"⚠️ {t}: Not enough data", parse_mode='HTML')
                continue

            # --- LOGIC ---
            df = run_vova_logic(df, scan_p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # Analyze Current
            valid, d, errs = analyze_trade(df, -1)
            
            # Check if New
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            # --- FILTERING LOGIC ---
            # Auto Mode: Показываем только valid
            # Manual Mode: Показываем ВСЕ (диагностика)
            
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
            
            if show_card:
                info = get_extended_info(t)
                risk_per_share = d['P'] - d['SL']
                shares = 0
                if risk_per_share > 0:
                    shares = int(scan_p['risk_usd'] / risk_per_share)
                
                card = format_dashboard_card(t, d, shares, is_new, info, scan_p['risk_usd'])
                await context.bot.send_message(chat_id=chat_id, text=card, parse_mode=constants.ParseMode.HTML, disable_web_page_preview=True)
                results_found += 1
            
        except Exception as e:
            if manual_mode:
                # await context.bot.send_message(chat_id, f"⚠️ {t}: Error {e}")
                pass

    global last_scan_time
    last_scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    final_txt = (
        f"🏁 <b>SCAN COMPLETE</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"✅ <b>Found:</b> {results_found}\n"
        f"📊 <b>Total:</b> {total}\n"
    )
    await context.bot.send_message(chat_id=chat_id, text=final_txt, parse_mode='HTML')
    context.user_data['scanning'] = False

# ==========================================
# 6. HELPER FUNCTIONS & HANDLERS
# ==========================================

def is_market_open():
    tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end

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
    if 'active_users' not in context.bot_data: context.bot_data['active_users'] = set()
    context.bot_data['active_users'].add(user_id)
    allowed = get_allowed_users()
    if user_id not in allowed:
        return False
    return True

async def safe_get_params(context):
    if 'params' not in context.user_data:
        context.user_data['params'] = DEFAULT_PARAMS.copy()
    else:
        current = context.user_data['params']
        new_params = DEFAULT_PARAMS.copy()
        new_params.update(current)
        context.user_data['params'] = new_params
    return context.user_data['params']

def get_reply_keyboard(p):
    risk_txt = f"💸 Risk: ${p['risk_usd']:.0f}"
    rr_txt = f"⚖️ RR: {p['min_rr']}"
    atr_txt = f"📊 ATR: {p['max_atr']}%"
    sma_txt = f"📈 SMA: {p['sma']}"
    tf_txt = "📅 Daily" if p['tf'] == 'Daily' else "🗓 Weekly"
    new_status = "✅" if p['new_only'] else "❌"
    new_txt = f"Only New signals {new_status}"
    
    keyboard = [
        [KeyboardButton(risk_txt), KeyboardButton(rr_txt)],
        [KeyboardButton(atr_txt), KeyboardButton(sma_txt)],
        [KeyboardButton(tf_txt), KeyboardButton(new_txt)], 
        [KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP SCAN")],
        [KeyboardButton("ℹ️ HELP / INFO")] 
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, is_persistent=True)

def get_status_text(status="💤 Idle", p=None):
    if not p: return f"Status: {status}"
    return (
        f"🖥 <b>Vova Screener Bot</b>\n━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ <b>Status:</b> {status}\n"
        f"🕒 <b>Last Scan:</b> {last_scan_time}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Config:</b> Risk <b>${p['risk_usd']}</b> (Min RR: {p['min_rr']})\n"
        f"🔍 <b>Filters:</b> {p['tf']} | SMA {p['sma']} | {'Only New' if p['new_only'] else 'All'}"
    )

def get_help_message():
    return (
        "📚 <b>CONFIGURATION GUIDE</b>\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "<b>💸 Risk $</b>: Max dollar loss per trade.\n"
        "<b>⚖️ RR</b>: Minimum Risk/Reward Ratio (e.g. 1.5).\n"
        "<b>📊 ATR %</b>: Max volatility allowed.\n"
        "<b>📈 SMA</b>: Trend filter (Price > SMA).\n"
        "<b>✨ Only New</b>: \n✅ = Show only fresh signals from TODAY.\n❌ = Show ALL valid signals found.\n"
        "<b>🔍 Manual Scan</b>: Send tickers comma separated (e.g. <code>AAPL, TSLA</code>) to see DIAGNOSIS."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    p = await safe_get_params(context)
    context.user_data['scanning'] = False
    context.user_data['input_mode'] = None
    
    welcome_txt = (
        f"👋 <b>Welcome, {update.effective_user.first_name}!</b>\n\n"
        f"💎 <b>Vova Screener Bot</b> is ready.\n"
        f"Tap 'Start Scan' to begin or send tickers manually."
    )
    await update.message.reply_html(welcome_txt, reply_markup=get_reply_keyboard(p))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    active = context.bot_data.get('active_users', set())
    msg = f"📊 <b>ADMIN STATS</b>\nActive: {len(active)}\nLast Scan: {last_scan_time}"
    await update.message.reply_html(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    
    text = update.message.text
    p = await safe_get_params(context)
    
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'): 
            await update.message.reply_text("⚠️ Scan already running!")
            return
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=False))
        return

    elif text == "⏹ STOP SCAN":
        context.user_data['scanning'] = False
        await update.message.reply_text("🛑 Stopping...")
        return

    elif text == "ℹ️ HELP / INFO":
        await update.message.reply_html(get_help_message())
        return

    elif "Daily" in text or "Weekly" in text:
        p['tf'] = "Weekly" if p['tf'] == "Daily" else "Daily"
    elif "Only New signals" in text:
        p['new_only'] = not p['new_only']

    elif "SMA:" in text:
        opts = [100, 150, 200]
        try: 
            current = int(text.split(":")[1].strip())
            p['sma'] = opts[(opts.index(current) + 1) % 3]
        except: p['sma'] = 200

    elif "Risk:" in text:
        context.user_data['input_mode'] = "risk_usd"
        await update.message.reply_text("✏️ Enter Risk Amount in $ (e.g., 50):")
        return
    elif "RR:" in text:
        context.user_data['input_mode'] = "min_rr"
        await update.message.reply_text("✏️ Enter Min RR (e.g., 2.0):")
        return
    elif "ATR:" in text:
        context.user_data['input_mode'] = "max_atr"
        await update.message.reply_text("✏️ Enter Max ATR % (e.g., 5.0):")
        return

    elif context.user_data.get('input_mode'):
        try:
            val = float(text.replace(',', '.'))
            mode = context.user_data['input_mode']
            if mode == "risk_usd": p['risk_usd'] = max(1.0, val)
            elif mode == "min_rr": p['min_rr'] = max(1.0, val)
            elif mode == "max_atr": p['max_atr'] = val
            context.user_data['input_mode'] = None
            await update.message.reply_text("✅ Updated!")
        except:
            await update.message.reply_text("❌ Invalid number. Try again.")
            return

    # MANUAL TICKER SCAN
    elif "," in text or (text.isalpha() and len(text) < 6):
        ts = [x.strip().upper() for x in text.split(",") if x.strip()]
        if ts:
            context.user_data['scanning'] = True
            await context.bot.send_message(update.effective_chat.id, f"🔎 Diagnosing: {ts}")
            # manual_mode=True включает показ ВСЕХ карточек (даже с ошибками)
            await run_scan_process(update, context, p, ts, manual_mode=True)
        return

    context.user_data['params'] = p
    await update.message.reply_text(get_status_text("Ready", p), reply_markup=get_reply_keyboard(p), parse_mode='HTML')

# ==========================================
# 7. MAIN (BACKGROUND RUN FIX)
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
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.create_task(application.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False, stop_signals=None))
        
        print("✅ Бот запущен в фоне")
    else:
        st.info("Бот активен и работает в фоне.")
