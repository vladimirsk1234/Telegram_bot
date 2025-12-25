import logging
import asyncio
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

# Suppress pandas fragmentation warnings
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
# 1. CONFIGURATION
# ==========================================
# REPLACE THESE WITH YOUR ACTUAL CREDENTIALS OR LOAD FROM ENV VARIABLES
TG_TOKEN = "YOUR_TOKEN_HERE" 
ADMIN_ID = 123456789 # YOUR_ID_HERE
GITHUB_USERS_URL = "" # Optional

# Strategy Constants (Matched to Pine Script)
EMA_F = 20
EMA_S = 40
ADX_L = 14
ADX_T = 20
ATR_L = 14
SMA_MAJ = 200

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default User State
DEFAULT_PARAMS = {
    'risk_usd': 100.0,
    'min_rr': 1.5,
    'max_atr': 5.0,
    'tf': 'Daily',
    'new_only': True,
}

# ==========================================
# 2. DATA & MATH (ALIGNED WITH TRADINGVIEW)
# ==========================================

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Failed to fetch SP500: {e}")
        return []

def get_fundamentals(ticker, close_price):
    """
    Replicates Pine Script P/E logic:
    pe_ttm = request.financial(..., "PRICE_EARNINGS_TTM", ...)
    pe_final = not na(pe_ttm) ? pe_ttm : (not na(eps_ttm) ? close / eps_ttm : na)
    """
    try:
        t = yf.Ticker(ticker)
        # fast_info is faster for Market Cap
        mc = t.fast_info.market_cap
        
        # We need .info for P/E and EPS
        info = t.info
        
        # 1. Try explicit Trailing PE
        pe = info.get('trailingPE')
        
        # 2. Fallback: Calculation via EPS (Pine Script Alignment)
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
    except Exception as e:
        logger.error(f"Fundamental error {ticker}: {e}")
        return {"mc": "N/A", "pe": "N/A"}

# --- INDICATORS (VECTORIZED) ---
def calc_ema(s, l): return s.ewm(span=l, adjust=False).mean()

def calc_macd_impulse(df, fast, slow, sig):
    ema_fast = calc_ema(df['Close'], fast)
    ema_slow = calc_ema(df['Close'], slow) # Center line
    
    # MACD Calculation
    macd_fast = calc_ema(df['Close'], 12)
    macd_slow = calc_ema(df['Close'], 26)
    macd_line = macd_fast - macd_slow
    signal_line = calc_ema(macd_line, 9)
    macd_hist = macd_line - signal_line
    
    # Elder Impulse Logic (Aligned with Pine)
    # Green: EMA13 rising AND MACD Hist rising
    bull = (ema_fast > ema_fast.shift(1)) & (macd_hist > macd_hist.shift(1))
    # Red: EMA13 falling AND MACD Hist falling
    bear = (ema_fast < ema_fast.shift(1)) & (macd_hist < macd_hist.shift(1))
    
    return bull, bear, ema_slow

def calc_adx(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    down = l.shift(1) - l
    
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Wilder's Smoothing (RMA) matches Pine Script 'rma'
    def rma(s, length): return s.ewm(alpha=1/length, adjust=False).mean()
    
    tr_s = rma(tr, length)
    p_di = 100 * rma(pd.Series(p_dm, index=df.index), length) / tr_s
    m_di = 100 * rma(pd.Series(m_dm, index=df.index), length) / tr_s
    
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di)
    adx = rma(dx, length)
    
    return adx, p_di, m_di

def run_vova_strategy(df):
    """
    Implements the core logic of 'sequence Vova' Pine Script
    """
    # 1. EMAs & MACD (Elder)
    bull_imp, bear_imp, ema_center = calc_macd_impulse(df, EMA_F, EMA_S, 9)
    sma_maj = df['Close'].rolling(SMA_MAJ).mean()
    
    # 2. ADX
    adx, p_di, m_di = calc_adx(df, ADX_L)
    
    # 3. ATR
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ATR_L, adjust=False).mean()
    
    # 4. EFI (Elder Force Index)
    efi = calc_ema(df['Close'].diff() * df['Volume'], EMA_F)
    
    # 5. VOVA STRUCTURE (Iterative Logic required for state)
    # This part must be iterative as it depends on previous sequence state
    n = len(df)
    c_arr = df['Close'].values
    h_arr = df['High'].values
    l_arr = df['Low'].values
    
    # Output arrays
    seq_state = np.zeros(n, dtype=int)
    crit_level = np.full(n, np.nan)
    struct_peak = np.full(n, np.nan)
    struct_valid = np.zeros(n, dtype=bool) # HH + HL check
    
    # State variables
    s_state = 0
    s_crit = np.nan
    s_h = h_arr[0]
    s_l = l_arr[0]
    last_confirmed_peak = np.nan
    last_confirmed_trough = np.nan
    last_peak_hh = False
    last_trough_hl = False
    
    for i in range(1, n):
        # Retrieve current bars
        close_i, high_i, low_i = c_arr[i], h_arr[i], l_arr[i]
        
        # Retrieve previous state
        prev_state = s_state
        prev_crit = s_crit
        prev_sh = s_h
        prev_sl = s_l
        
        is_break = False
        
        # Check Break
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = close_i < prev_crit
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_break = close_i > prev_crit
            
        if is_break:
            if prev_state == 1: # Up Trend Broken (Bearish)
                # Logic for HH check
                is_hh = True if np.isnan(last_confirmed_peak) else (prev_sh > last_confirmed_peak)
                last_peak_hh = is_hh
                last_confirmed_peak = prev_sh
                
                # Switch to Down
                s_state = -1
                s_h = high_i
                s_l = low_i
                s_crit = high_i
            else: # Down Trend Broken (Bullish)
                # Logic for HL check
                is_hl = True if np.isnan(last_confirmed_trough) else (prev_sl > last_confirmed_trough)
                last_trough_hl = is_hl
                last_confirmed_trough = prev_sl
                
                # Switch to Up
                s_state = 1
                s_h = high_i
                s_l = low_i
                s_crit = low_i
        else:
            s_state = prev_state
            if s_state == 1:
                if high_i >= s_h: s_h = high_i
                # Trail Logic
                if high_i >= prev_sh: s_crit = low_i
                else: s_crit = prev_crit
            elif s_state == -1:
                if low_i <= s_l: s_l = low_i
                # Trail Logic
                if low_i <= prev_sl: s_crit = high_i
                else: s_crit = prev_crit
            else:
                # Initialization
                if close_i > prev_sh: 
                    s_state = 1; s_crit = low_i
                elif close_i < prev_sl: 
                    s_state = -1; s_crit = high_i
                else:
                    s_h = max(prev_sh, high_i)
                    s_l = min(prev_sl, low_i)
        
        seq_state[i] = s_state
        crit_level[i] = s_crit
        struct_peak[i] = last_confirmed_peak
        # The crucial check: Last Confirmed Peak was HH AND Last Confirmed Trough was HL
        struct_valid[i] = last_peak_hh and last_trough_hl

    # 6. COMBINED SUPER TREND LOGIC
    # Strict Green: ADX confirms + Impulse Bull + EFI confirms
    adx_bull = (adx >= ADX_T) & (p_di > m_di)
    adx_bear = (adx >= ADX_T) & (m_di > p_di)
    
    efi_bull = efi > 0
    efi_bear = efi < 0
    
    trend_state = np.zeros(n, dtype=int)
    
    is_bull = adx_bull & bull_imp & efi_bull
    is_bear = adx_bear & bear_imp & efi_bear
    
    trend_state[is_bull] = 1
    trend_state[is_bear] = -1
    
    # Store results in DF
    df['Seq'] = seq_state
    df['Crit'] = crit_level
    df['Peak'] = struct_peak
    df['StructOk'] = struct_valid
    df['Trend'] = trend_state
    df['ATR'] = atr
    df['SMA'] = sma_maj
    
    return df

# ==========================================
# 3. TELEGRAM UI & FORMATTING (VISUAL FIX)
# ==========================================

def format_card(ticker, row, shares, is_new, funds, risk_usd):
    """
    Creates a clean, card-like output without white lines.
    Uses HTML for bolding and code blocks for alignment.
    """
    price = row['Close']
    sl = row['Crit'] # Structural SL
    atr_sl = price - row['ATR']
    final_sl = min(sl, atr_sl) if not np.isnan(sl) else atr_sl
    
    tp = row['Peak']
    risk = price - final_sl
    reward = tp - price
    rr = reward / risk if risk > 0 else 0
    
    # Formatting Emojis
    atr_pct = (row['ATR'] / price) * 100
    atr_dot = "🔴" if atr_pct > 5.0 else ("🟡" if atr_pct >= 3.0 else "🟢")
    
    trend_dot = {1: "🟢", -1: "🔴", 0: "🟡"}.get(row['Trend'], "🟡")
    seq_dot = {1: "🟢", -1: "🔴", 0: "🟡"}.get(row['Seq'], "🟡")
    ma_dot = "🟢" if price > row['SMA'] else "🔴"
    
    # Status Header
    status_icon = "🆕" if is_new else "♻️"
    
    # TradingView Link
    tv_url = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    
    # Card Construction
    # 1. Header line (Ticker + Price + Status)
    card = f"<b><a href='{tv_url}'>{ticker}</a></b>   <code>${price:.2f}</code>   {status_icon}\n"
    
    # 2. Fundamentals & ATR (Sub-header)
    card += f"PE: <code>{funds['pe']}</code> | MC: <code>{funds['mc']}</code>\n"
    card += f"ATR: <code>{atr_pct:.2f}%</code> {atr_dot}\n\n"
    
    # 3. Technical Grid (No Separators)
    card += f"<b>Trend</b> {trend_dot}  <b>Seq</b> {seq_dot}  <b>MA</b> {ma_dot}\n\n"
    
    # 4. Trade Setup (Monospace block for alignment)
    profit = (tp - price) * shares
    loss_amt = (price - final_sl) * shares
    
    card += f"🎯 TP: <code>{tp:.2f}</code> (<i>+${profit:.0f}</i>)\n"
    card += f"🛑 SL: <code>{final_sl:.2f}</code> (<i>-${loss_amt:.0f}</i>)\n"
    card += f"⚖️ RR: <code>{rr:.2f}</code>  📦 Size: <code>{shares}</code>"
    
    return card

# ==========================================
# 4. BOT COMMANDS & LOGIC
# ==========================================

async def run_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    params = context.user_data.get('params', DEFAULT_PARAMS)
    
    status_msg = await context.bot.send_message(
        chat_id, 
        "⏳ <b>Initializing Scan...</b>\nFetching S&P 500 list.", 
        parse_mode=ParseMode.HTML
    )
    
    tickers = get_sp500_tickers()
    if not tickers:
        await status_msg.edit_text("❌ Error fetching tickers.")
        return

    # User control flag
    context.user_data['scanning'] = True
    
    total = len(tickers)
    found = 0
    
    # Timeframe selection
    tf_map = {'Daily': '1d', 'Weekly': '1wk'}
    period_map = {'Daily': '2y', 'Weekly': '5y'}
    
    interval = tf_map[params['tf']]
    period = period_map[params['tf']]
    
    # MAIN LOOP
    for i, ticker in enumerate(tickers):
        # Stop check
        if not context.user_data.get('scanning'):
            await context.bot.send_message(chat_id, "⏹ Scan Aborted.")
            break
            
        # Progress Update (Every 20 tickers)
        if i % 20 == 0:
            pct = int((i / total) * 100)
            await status_msg.edit_text(f"🔍 <b>Scanning...</b> {pct}%\nChecking: {ticker}", parse_mode=ParseMode.HTML)

        try:
            # 1. Fetch Price Data (Async-friendly delay)
            await asyncio.sleep(0.05) 
            df = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
            
            if len(df) < SMA_MAJ + 5: continue
            
            # 2. Run Strategy
            df = run_vova_strategy(df)
            
            # 3. Check Signal (Last closed bar)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Conditions
            cond_seq = curr['Seq'] == 1
            cond_ma = curr['Close'] > curr['SMA']
            cond_trend = curr['Trend'] != -1
            cond_struct = curr['StructOk'] # HH/HL check
            
            is_valid = cond_seq and cond_ma and cond_trend and cond_struct
            
            # Additional Filters
            sl = curr['Crit']
            atr_sl = curr['Close'] - curr['ATR']
            final_sl = min(sl, atr_sl) if not np.isnan(sl) else atr_sl
            risk = curr['Close'] - final_sl
            reward = curr['Peak'] - curr['Close']
            
            rr = reward / risk if risk > 0 else 0
            atr_pct = (curr['ATR'] / curr['Close']) * 100
            
            # Filter Checks
            if is_valid:
                if rr < params['min_rr']: continue
                if atr_pct > params['max_atr']: continue
                
                # New Signal Only Check
                is_new = False
                prev_valid = (prev['Seq'] == 1 and prev['Close'] > prev['SMA'] and prev['Trend'] != -1 and prev['StructOk'])
                if not prev_valid: is_new = True
                
                if params['new_only'] and not is_new: continue
                
                # 4. Calculate Position
                shares = int(params['risk_usd'] / risk)
                if shares < 1: continue
                
                # 5. Get Fundamentals (Lazy Load - Only for Hits)
                funds = get_fundamentals(ticker, curr['Close'])
                
                # 6. Send Card
                card = format_card(ticker, curr, shares, is_new, funds, params['risk_usd'])
                await context.bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                found += 1
                
        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
            continue

    await status_msg.edit_text(f"✅ <b>Scan Complete</b>\nFound: {found} tickers.")
    context.user_data['scanning'] = False

# --- HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != ADMIN_ID: return # Simple Auth
    
    if 'params' not in context.user_data:
        context.user_data['params'] = DEFAULT_PARAMS.copy()
        
    kb = [
        [KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP")],
        [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ Status")]
    ]
    await update.message.reply_text("🤖 <b>QuantBot V2 Ready</b>", 
                                    reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True), 
                                    parse_mode=ParseMode.HTML)

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'):
            await update.message.reply_text("⚠️ Scan already running.")
        else:
            asyncio.create_task(run_scan(update, context))
    elif text == "⏹ STOP":
        context.user_data['scanning'] = False
        await update.message.reply_text("🛑 Stopping scan...")
    elif text == "⚙️ Settings":
        p = context.user_data.get('params', DEFAULT_PARAMS)
        msg = (f"<b>Current Settings:</b>\n"
               f"Risk: ${p['risk_usd']}\n"
               f"Min RR: {p['min_rr']}\n"
               f"Timeframe: {p['tf']}")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

if __name__ == '__main__':
    # Initialize Application
    app = ApplicationBuilder().token(TG_TOKEN).persistence(PicklePersistence("bot_data.pickle")).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    
    print("🚀 Bot is running. Press Ctrl+C to stop.")
    app.run_polling()
