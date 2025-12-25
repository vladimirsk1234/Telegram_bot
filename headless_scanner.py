import logging
import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, PicklePersistence

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
TG_TOKEN = "YOUR_TOKEN_HERE"  # Replace or load from env
ADMIN_ID = 123456789          # Replace with your ID

# Strategy Settings
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14; SMA_MAJ = 200

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- MATH & LOGIC ---

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Random User-Agent to prevent 403 Forbidden
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Error fetching SP500: {e}")
        return []

def get_fundamentals(ticker, close_price):
    """
    Robust fetch using fast_info where possible to avoid blocking scraping.
    """
    try:
        t = yf.Ticker(ticker)
        # fast_info is non-blocking (calculated locally or lightweight fetch)
        mc = t.fast_info.market_cap
        
        # info is blocking/scraping, but we need it for PE. 
        # We will run this function in an executor later.
        info = t.info
        pe = info.get('trailingPE')
        
        # Fallback P/E calculation
        if pe is None:
            eps = info.get('trailingEps')
            if eps: pe = close_price / eps

        # Format Market Cap
        if mc >= 1e12: mc_str = f"{mc/1e12:.2f}T"
        elif mc >= 1e9: mc_str = f"{mc/1e9:.2f}B"
        else: mc_str = f"{mc/1e6:.2f}M"
        
        pe_str = f"{pe:.2f}" if pe else "N/A"
        return {"mc": mc_str, "pe": pe_str}
    except:
        return {"mc": "N/A", "pe": "N/A"}

# [Insert your calc_ema, calc_adx_pine, calc_atr, run_vova_logic, analyze_trade functions here]
# ... (Keeping your existing logic functions as they were mostly math-correct) ...
# For brevity, I am assuming the math functions are pasted here.

# --- ASYNC WRAPPERS (CRITICAL FIX) ---

async def fetch_data_async(ticker, period, interval):
    """
    Wraps blocking yfinance call in an executor to keep bot responsive.
    """
    loop = asyncio.get_running_loop()
    try:
        # run_in_executor(None, ...) uses the default ThreadPoolExecutor
        df = await loop.run_in_executor(None, lambda: yf.download(
            ticker, period=period, interval=interval, progress=False, 
            auto_adjust=False, multi_level_index=False
        ))
        return df
    except Exception:
        return pd.DataFrame()

async def fetch_fundamentals_async(ticker, close):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: get_fundamentals(ticker, close))

# --- UI ---

def format_card(ticker, d, shares, is_new, funds):
    status_icon = "🆕" if is_new else "♻️"
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    
    # Clean HTML Layout
    return (
        f"<b><a href='{tv_link}'>{ticker}</a></b> {status_icon}\n"
        f"<code>Price: ${d['P']:.2f}</code>\n"
        f"<code>PE: {funds['pe']} | MC: {funds['mc']}</code>\n\n"
        f"🎯 <b>TP:</b> <code>{d['TP']:.2f}</code>\n"
        f"🛑 <b>SL:</b> <code>{d['SL']:.2f}</code>\n"
        f"⚖️ <b>R/R:</b> <code>{d['RR']:.2f}</code>  📦 <b>Size:</b> <code>{shares}</code>"
    )

# --- SCANNER ENGINE ---

async def scanner_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    tickers = get_sp500_tickers() # This blocks briefly, acceptable for start
    
    if not tickers:
        await context.bot.send_message(chat_id, "❌ Failed to fetch tickers.")
        return

    status_msg = await context.bot.send_message(chat_id, f"🔎 <b>Scanning {len(tickers)} tickers...</b>", parse_mode=ParseMode.HTML)
    
    context.user_data['scanning'] = True
    found = 0
    
    for i, t in enumerate(tickers):
        # 1. Check Cancellation
        if not context.user_data.get('scanning'):
            await status_msg.edit_text("⏹ <b>Scan Stopped.</b>", parse_mode=ParseMode.HTML)
            break
            
        # 2. Update UI (Every 20 tickers to avoid rate limits)
        if i % 20 == 0:
            try:
                await status_msg.edit_text(f"🔎 Scanning... {int((i/len(tickers))*100)}% ({t})", parse_mode=ParseMode.HTML)
            except: pass # Ignore "message not modified" errors

        # 3. Non-Blocking Fetch
        df = await fetch_data_async(t, "2y", "1d")
        
        if len(df) < SMA_MAJ + 5: continue
        
        # 4. Logic (CPU bound, fast enough to run in loop)
        # Assuming run_vova_logic is defined
        # df = run_vova_logic(df, ...) 
        # valid, d, _ = analyze_trade(df, -1)
        
        # [Placeholder for your strategy logic integration]
        # For demonstration, let's assume we found a valid trade:
        valid = False # Set to True to test
        
        if valid:
            # Non-blocking fundamentals fetch
            funds = await fetch_fundamentals_async(t, d['P'])
            card = format_card(t, d, 100, True, funds)
            await context.bot.send_message(chat_id, card, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            found += 1
            
    if context.user_data.get('scanning'):
        await status_msg.edit_text(f"✅ <b>Scan Complete.</b> Found: {found}", parse_mode=ParseMode.HTML)

# --- HANDLERS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    kb = [[KeyboardButton("▶️ START SCAN"), KeyboardButton("⏹ STOP SCAN")]]
    await update.message.reply_text("Bot Ready.", reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    text = update.message.text
    
    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'):
            await update.message.reply_text("⚠️ Scan already running.")
        else:
            # Run scanner as a background task so it doesn't block the bot listener
            asyncio.create_task(scanner_task(update, context))
            
    elif text == "⏹ STOP SCAN":
        context.user_data['scanning'] = False
        await update.message.reply_text("🛑 Stopping...")

if __name__ == '__main__':
    # Initialize Application
    app = ApplicationBuilder().token(TG_TOKEN).persistence(PicklePersistence("bot_data.pickle")).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, handle_msg))
    
    print("🚀 Bot is running...")
    app.run_polling()
