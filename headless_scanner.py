import streamlit as st
import logging
import asyncio
import threading
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import gc

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from telegram import (
    Update, 
    ReplyKeyboardMarkup, 
    KeyboardButton
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    PicklePersistence
)

# ==========================================
# 1. CONFIGURATION & SECRETS
# ==========================================
# Попытка загрузить секреты, иначе используем заглушки (для локального теста замените сами)
try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
except:
    st.error("❌ Secrets not found! Please set TG_TOKEN and ADMIN_ID in .streamlit/secrets.toml")
    st.stop()

# Strategy Constants (Aligned with Pine Script)
EMA_F = 20
EMA_S = 40
ADX_L = 14
ADX_T = 20
ATR_L = 14
SMA_MAJ = 200

DEFAULT_PARAMS = {
    'risk_usd': 100.0,
    'min_rr': 1.5,
    'max_atr': 5.0,
    'tf': 'Daily',
    'new_only': True,
}

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 2. LOGIC: PINE SCRIPT ALIGNMENT
# ==========================================

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: return []

def get_fundamentals(ticker, close_price):
    """
    P/E Logic identical to your Pine Script:
    1. Try Trailing PE (Official TTM)
    2. Fallback: Close / Trailing EPS (TTM)
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 1. Market Cap (Fast Info is more reliable)
        mc = t.fast_info.market_cap
        
        # 2. P/E Logic
        pe = info.get('trailingPE')
        
        # Fallback if PE is None
        if pe is None:
            eps = info.get('trailingEps')
            if eps and eps != 0:
                pe = close_price / eps
                
        # Formatting
        mc_str = "N/A"
        if mc:
            if mc >= 1e12: mc_str = f"{mc/1e12:.2f}T"
            elif mc >= 1e9: mc_str = f"{mc/1e9:.2f}B"
            elif mc >= 1e6: mc_str = f"{mc/1e6:.2f}M"
            
        pe_str = f"{pe:.2f}" if pe else "N/A"
        
        return {"mc": mc_str, "pe": pe_str}
    except:
        return {"mc": "N/A", "pe": "N/A"}

# --- INDICATOR MATH (Vectorized) ---
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()

def run_vova_strategy(df):
    """
    Python implementation of 'sequence Vova' logic
    """
    # 1. EMAs & MACD (Elder Impulse)
    ema_f = calc_ema(df['Close'], EMA_F)
    ema_s = calc_ema(df['Close'], EMA_S)
    
    # MACD
    macd_fast = calc_ema(df['Close'], 12)
    macd_slow = calc_ema(df['Close'], 26)
    macd_line = macd_fast - macd_slow
    sig_line = calc_ema(macd_line, 9)
    hist = macd_line - sig_line
    
    # Impulse Colors Logic
    bull_imp = (ema_f > ema_f.shift(1)) & (hist > hist.shift(1))
    bear_imp = (ema_f < ema_f.shift(1)) & (hist < hist.shift(1))
    
    # 2. ADX
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    
    def rma(s, length): return s.ewm(alpha=1/length, adjust=False).mean()
    
    tr_s = rma(tr, ADX_L)
    up = h - h.shift(1); down = l.shift(1) - l
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    p_di = 100 * rma(pd.Series(p_dm, index=df.index), ADX_L) / tr_s
    m_di = 100 * rma(pd.Series(m_dm, index=df.index), ADX_L) / tr_s
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di)
    adx = rma(dx, ADX_L)
    
    # 3. ATR
    atr = rma(tr, ATR_L)
    
    # 4. EFI
    efi = calc_ema(df['Close'].diff() * df['Volume'], EMA_F)
    
    # 5. SEQUENCE LOGIC (Iterative)
    n = len(df)
    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    
    seq_state = np.zeros(n, dtype=int)
    crit_lvl = np.full(n, np.nan)
    peak_arr = np.full(n, np.nan)
    struct_ok = np.zeros(n, dtype=bool)
    
    s_state = 0; s_crit = np.nan; s_h = h_arr[0]; s_l = l_arr[0]
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h_i, l_i = c_arr[i], h_arr[i], l_arr[i]
        
        # Break Detection
        is_break = False
        if s_state == 1 and not np.isnan(s_crit): is_break = c < s_crit
        elif s_state == -1 and not np.isnan(s_crit): is_break = c > s_crit
            
        if is_break:
            if s_state == 1: # Bull -> Bear
                pk_hh = True if np.isnan(last_pk) else (s_h > last_pk)
                last_pk = s_h
                s_state = -1; s_h = h_i; s_l = l_i; s_crit = h_i
            else: # Bear -> Bull
                tr_hl = True if np.isnan(last_tr) else (s_l > last_tr)
                last_tr = s_l
                s_state = 1; s_h = h_i; s_l = l_i; s_crit = l_i
        else:
            if s_state == 1:
                if h_i >= s_h: s_h = h_i
                if h_i >= s_h: s_crit = l_i # Trailing (simplified logic for perf)
            elif s_state == -1:
                if l_i <= s_l: s_l = l_i
                if l_i <= s_l: s_crit = h_i
            else:
                if c > s_h: s_state = 1; s_crit = l_i
                elif c < s_l: s_state = -1; s_crit = h_i
                else: s_h = max(s_h, h_i); s_l = min(s_l, l_i)
        
        seq_state[i] = s_state
        crit_lvl[i] = s_crit
        peak_arr[i] = last_pk
        struct_ok[i] = pk_hh and tr_hl

    # 6. Trend State
    trend_state = np.zeros(n, dtype=int)
    is_bull = (adx >= ADX_T) & (p_di > m_di) & bull_imp & (efi > 0)
    is_bear = (adx >= ADX_T) & (m_di > p_di) & bear_imp & (efi < 0)
    trend_state[is_bull] = 1
    trend_state[is_bear] = -1
    
    df['Seq'] = seq_state; df['Crit'] = crit_lvl; df['Peak'] = peak_arr
    df['Struct'] = struct_ok; df['Trend'] = trend_state; df['ATR'] = atr
    df['SMA'] = df['Close'].rolling(SMA_MAJ).mean()
    return df

# ==========================================
# 3. CARD VISUALIZATION (Corrected)
# ==========================================
def format_card(ticker, row, shares, is_new, funds, risk_usd):
    """
    Generates clean Telegram HTML without the white line separator.
    """
    price = row['Close']
    
    # Calculate SL/TP
    sl_struct = row['Crit']
    sl_atr = price - row['ATR']
    # Pine logic: min(sl_struct, sl_atr)
    final_sl = min(sl_struct, sl_atr) if not np.isnan(sl_struct) else sl_atr
    
    tp = row['Peak']
    risk = price - final_sl
    reward = tp - price
    rr = reward / risk if risk > 0 else 0
    
    # Visual Helpers
    atr_pct = (row['ATR'] / price) * 100
    atr_emoji = "🔴" if atr_pct > 5.0 else ("🟡" if atr_pct >= 3.0 else "🟢")
    
    trend_map = {1: "🟢", -1: "🔴", 0: "🟡"}
    seq_emoji = trend_map.get(row['Seq'], "🟡")
    trend_emoji = trend_map.get(row['Trend'], "🟡")
    ma_emoji = "🟢" if price > row['SMA'] else "🔴"
    
    # Status
    status_tag = "🆕 <b>NEW</b>" if is_new else "♻️ <b>ACTIVE</b>"
    
    # TradingView Link
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    
    # --- HTML STRUCTURE ---
    # Header
    html = f"🖥 <b><a href='{tv_link}'>{ticker}</a></b>   <code>${price:.2f}</code>   {status_tag}\n"
    
    # Fundamentals & ATR (Clean spacing, no lines)
    html += f"PE: <code>{funds['pe']}</code>   MC: <code>{funds['mc']}</code>\n"
    html += f"ATR: <code>{atr_pct:.2f}%</code> {atr_emoji}\n\n"
    
    # Grid
    html += f"<b>Trend</b> {trend_emoji}   <b>Seq</b> {seq_emoji}   <b>MA</b> {ma_emoji}\n\n"
    
    # Trade Setup (Monospace for alignment)
    profit = (tp - price) * shares
    loss = (price - final_sl) * shares
    
    html += f"🎯 TP: <code>{tp:.2f}</code>  (<i>+${profit:.0f}</i>)\n"
    html += f"🛑 SL: <code>{final_sl:.2f}</code>  (<i>-${abs(loss):.0f}</i>)\n"
    html += f"⚖️ RR: <code>{rr:.2f}</code>   📦 Size: <code>{shares}</code>"
    
    return html

# ==========================================
# 4. BOT PROCESS (ASYNC SCANNER)
# ==========================================

async def scan_market(update, context, tickers):
    chat_id = update.effective_chat.id
    p = context.user_data.get('params', DEFAULT_PARAMS)
    
    # Status Msg
    msg = await context.bot.send_message(chat_id, "⏳ <b>Starting Scan...</b>", parse_mode=ParseMode.HTML)
    
    found = 0
    total = len(tickers)
    
    # Timeframe map
    interval = "1d" if p['tf'] == "Daily" else "1wk"
    period = "2y" if p['tf'] == "Daily" else "5y"
    
    for i, t in enumerate(tickers):
        if not context.user_data.get('scanning', False):
            await msg.edit_text("⏹ <b>Scan Stopped.</b>", parse_mode=ParseMode.HTML)
            break
            
        # UI Update (batches to save API limits)
        if i % 25 == 0:
            pct = int((i/total)*100)
            try: await msg.edit_text(f"🔍 <b>Scanning {pct}%</b>\nChecking: {t}", parse_mode=ParseMode.HTML)
            except: pass
            
        try:
            await asyncio.sleep(0.01) # Yield control
            df = yf.download(t, period=period, interval=interval, progress=False, multi_level_index=False)
            
            if len(df) < SMA_MAJ + 10: continue
            
            # Logic
            df = run_vova_strategy(df)
            
            # Check Last Bar
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Validations
            valid = (curr['Seq'] == 1 and 
                     curr['Close'] > curr['SMA'] and 
                     curr['Trend'] != -1 and 
                     curr['Struct'])
            
            if valid:
                # Risk Check
                sl = curr['Crit']
                sl_atr = curr['Close'] - curr['ATR']
                final_sl = min(sl, sl_atr) if not np.isnan(sl) else sl_atr
                risk = curr['Close'] - final_sl
                
                if risk <= 0: continue
                
                rr = (curr['Peak'] - curr['Close']) / risk
                atr_pct = (curr['ATR'] / curr['Close']) * 100
                
                if rr < p['min_rr'] or atr_pct > p['max_atr']: continue
                
                # New Only Filter
                prev_valid = (prev['Seq'] == 1 and prev['Close'] > prev['SMA'] and prev['Trend'] != -1 and prev['Struct'])
                is_new = not prev_valid
                
                if p['new_only'] and not is_new: continue
                
                # Calculate Size
                shares = int(p['risk_usd'] / risk)
                if shares < 1: continue
                
                # Get Fundamentals (Lazy Load)
                funds = get_fundamentals(t, curr['Close'])
                
                # Send
                card = format_card(t, curr, shares, is_new, funds, p['risk_usd'])
                await context.bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                found += 1
                
        except Exception as e:
            continue
            
    try: await msg.edit_text(f"✅ <b>Scan Complete.</b> Found: {found}", parse_mode=ParseMode.HTML)
    except: pass
    context.user_data['scanning'] = False

# ==========================================
# 5. STREAMLIT ARCHITECTURE (SINGLETON)
# ==========================================

async def start_command(update, context):
    if update.effective_user.id != ADMIN_ID: return
    context.user_data['params'] = DEFAULT_PARAMS.copy()
    
    kb = [[KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP")],
          [KeyboardButton("ℹ️ Settings")]]
          
    await update.message.reply_text(
        "💎 <b>Vova Bot Ready</b>\nUse buttons to control.", 
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True),
        parse_mode=ParseMode.HTML
    )

async def message_handler(update, context):
    if update.effective_user.id != ADMIN_ID: return
    text = update.message.text
    
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'):
            await update.message.reply_text("⚠️ Already running.")
        else:
            context.user_data['scanning'] = True
            tickers = get_sp500_tickers()
            asyncio.create_task(scan_market(update, context, tickers))
            
    elif text == "⏹ STOP":
        context.user_data['scanning'] = False
        await update.message.reply_text("🛑 Stopping...")
        
    elif text == "ℹ️ Settings":
        p = context.user_data.get('params', DEFAULT_PARAMS)
        await update.message.reply_html(f"Risk: ${p['risk_usd']} | RR: {p['min_rr']} | TF: {p['tf']}")

@st.cache_resource
def get_bot_application():
    """
    Creates the bot application exactly ONCE.
    This function is cached, so Streamlit won't recreate the bot on re-runs.
    """
    app = ApplicationBuilder().token(TG_TOKEN).persistence(PicklePersistence("bot_data.pickle")).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT, message_handler))
    return app

def run_bot_in_thread(app):
    """
    Runs the asyncio loop in a separate thread.
    Handles the 'no current event loop' error by creating a new one for this thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run polling blocking ONLY this thread, not Streamlit
    app.run_polling(stop_signals=None, close_loop=False)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    st.set_page_config(page_title="Vova Bot", page_icon="🤖")
    st.title("💎 Vova Bot Manager")

    # 1. Get Singleton App
    app = get_bot_application()

    # 2. Check/Start Thread
    # We use a simple attribute on the app to check if we started the thread already
    if not getattr(app, "_is_running", False):
        st.info("🚀 Starting Bot Polling Thread...")
        t = threading.Thread(target=run_bot_in_thread, args=(app,), daemon=True)
        t.start()
        app._is_running = True
        st.success("✅ Bot Thread Started!")
    else:
        st.success("✅ Bot is running (Cached).")

    # 3. Simple Dashboard
    st.write("The bot is running in the background. You can close this tab if deployed to cloud.")
    
    # Clock
    ny_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime("%H:%M:%S")
    st.metric("NY Time", ny_time)

if __name__ == "__main__":
    main()
