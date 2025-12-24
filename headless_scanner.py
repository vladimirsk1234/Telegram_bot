import logging
import asyncio
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import nest_asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# --- KONFIGURACIJA ---
nest_asyncio.apply()

# ⚠️ ВСТАВЬТЕ СВОИ ДАННЫЕ СЮДА ⚠️
TG_TOKEN = "ВАШ_ТОКЕН_БОТА"
ADMIN_ID = 123456789  # Ваш ID (число)
GITHUB_USERS_URL = "ССЫЛКА_НА_RAW_СПИСОК_ЮЗЕРОВ" # Пример: raw.githubusercontent...

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальные константы логики (скрытые)
EMA_F = 20
EMA_S = 40
ADX_L = 14
ADX_T = 20
ATR_L = 14

# ==========================================
# 1. ЛОГИКА СКРИНЕРА (100% COPY FROM WEB)
# ==========================================

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Error S&P500: {e}")
        return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# --- INDICATOR MATH ---
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

# --- VOVA STRATEGY LOGIC ---
def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    # --- Indicators ---
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    # --- Iterative Structure Logic ---
    n = len(df)
    c_a, h_a, l_a = df['Close'].values, df['High'].values, df['Low'].values
    
    seq_st = np.zeros(n, dtype=int)
    crit_lvl = np.full(n, np.nan)
    res_peak = np.full(n, np.nan)
    res_struct = np.zeros(n, dtype=bool)
    
    # State Variables
    s_state = 0
    s_crit = np.nan
    s_h = h_a[0]; s_l = l_a[0]
    
    last_pk = np.nan; last_tr = np.nan
    pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        
        # Previous values
        prev_st = s_state
        prev_cr = s_crit
        prev_sh = s_h
        prev_sl = s_l
        
        # Break Detection
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
            
        if brk:
            if prev_st == 1: # Bearish Break
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                pk_hh = is_hh
                last_pk = prev_sh
                s_state = -1
                s_h = h; s_l = l
                s_crit = h
            else: # Bullish Break
                is_hl = True if np.isnan(last_tr) else (prev_sl > last_tr)
                tr_hl = is_hl
                last_tr = prev_sl
                s_state = 1
                s_h = h; s_l = l
                s_crit = l
        else:
            s_state = prev_st
            
            if s_state == 1: # Uptrend
                if h >= s_h: s_h = h
                if h >= prev_sh: s_crit = l
                else: s_crit = prev_cr
                
            elif s_state == -1: # Downtrend
                if l <= s_l: s_l = l
                if l <= prev_sl: s_crit = h
                else: s_crit = prev_cr
                
            else: # Init
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

    # --- Super Trend Logic ---
    adx_str = adx >= adx_thr
    
    bull = (adx_str & (p_di > m_di)) & \
           ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & \
           (efi > 0)
           
    bear = (adx_str & (m_di > p_di)) & \
           ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & \
           (efi < 0)
           
    t_st = np.zeros(n, dtype=int)
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
    errs = []
    
    if r['Seq'] != 1: errs.append("SEQ!=1")
    if np.isnan(r['SMA']) or r['Close'] <= r['SMA']: errs.append("SMA")
    if r['Trend'] == -1: errs.append("TREND")
    if not r['Struct']: errs.append("STRUCT")
    if np.isnan(r['Peak']) or np.isnan(r['Crit']): errs.append("NO DATA")
    
    if errs: return False, {}, " ".join(errs)
    
    price = r['Close']
    tp = r['Peak']
    crit = r['Crit']
    atr = r['ATR']
    
    sl_struct = crit
    sl_atr = price - atr
    final_sl = min(sl_struct, sl_atr)
    
    risk = price - final_sl
    reward = tp - price
    
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    rr = reward / risk
    
    return True, {
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": rr, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"

# ==========================================
# 3. HELPER FUNCTIONS & UI FORMATTING
# ==========================================

def is_market_open():
    """Проверка времени работы рынка США (9:30 - 16:00 ET, будни)"""
    tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return False # Выходные
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end

def format_luxury_card(ticker, d, shares, is_new, pe_val):
    """Создает компактную LUXURY карточку для Telegram (HTML)"""
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    
    # Emojis & Formatting
    badge = "🆕" if is_new else ""
    pe_str = f"| P/E: <b>{pe_val:.0f}</b>" if pe_val else ""
    
    val_pos = shares * d['P']
    profit_pot = (d['TP'] - d['P']) * shares
    loss_pot = (d['P'] - d['SL']) * shares
    atr_pct = (d['ATR'] / d['P']) * 100
    
    card = (
        f"💎 <b><a href='{tv_link}'>{ticker}</a></b> {badge}\n"
        f"💵 <b>{d['P']:.2f}</b> {pe_str}\n"
        f"💼 <b>POS:</b> {shares} (<b>${val_pos:.0f}</b>) | ⚖️ <b>R:R:</b> {d['RR']:.2f}\n"
        f"🎯 <b>TP:</b> {d['TP']:.2f} (<span class='tg-spoiler'>+${profit_pot:.0f}</span>)\n"
        f"🛑 <b>SL:</b> {d['SL']:.2f} (<span class='tg-spoiler'>-${loss_pot:.0f}</span>) [{d['SL_Type']}]\n"
        f"📉 <b>Crit:</b> {d['Crit']:.2f}\n"
        f"📊 <b>ATR:</b> {d['ATR']:.2f} ({atr_pct:.1f}%)"
    )
    return card

# ==========================================
# 4. TELEGRAM BOT HANDLERS & STATE
# ==========================================

# Инициализация дефолтных параметров пользователя
DEFAULT_PARAMS = {
    'portfolio': 10000.0,
    'min_rr': 1.25,
    'risk_pct': 0.2,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
    'autoscan': False,
    'mode': 'SP500' # 'SP500' or 'MANUAL'
}

# Кэш отправленных сегодня тикеров (для автоскана)
sent_today = set()
last_scan_time = "Никогда"

def get_keyboard(p):
    """Генерация меню с параметрами (значения на кнопках)"""
    # Индикаторы состояния
    tf_txt = "📅 D1" if p['tf'] == 'Daily' else "📅 W1"
    new_txt = "🆕 On" if p['new_only'] else "🆕 Off"
    auto_txt = "🤖 On" if p['autoscan'] else "🤖 Off"
    
    kb = [
        [
            InlineKeyboardButton(f"💰 ${p['portfolio']:.0f}", callback_data="set_port"),
            InlineKeyboardButton(f"⚖️ RR: {p['min_rr']}", callback_data="set_rr"),
        ],
        [
            InlineKeyboardButton(f"⚠️ Risk: {p['risk_pct']}%", callback_data="set_risk"),
            InlineKeyboardButton(f"📊 ATR: {p['max_atr']}%", callback_data="set_matr"),
        ],
        [
            InlineKeyboardButton(f"📈 SMA {p['sma']}", callback_data="set_sma"),
            InlineKeyboardButton(tf_txt, callback_data="toggle_tf"),
            InlineKeyboardButton(new_txt, callback_data="toggle_new"),
        ],
        [
            InlineKeyboardButton("▶️ START SCAN", callback_data="start_scan"),
            InlineKeyboardButton("⏹ STOP", callback_data="stop_scan"),
        ],
        [
            InlineKeyboardButton(f"AutoScan: {auto_txt}", callback_data="toggle_auto"),
        ]
    ]
    return InlineKeyboardMarkup(kb)

def get_status_text(status="💤 Ожидание", p=None):
    if not p: return f"Статус: {status}"
    return (
        f"🖥 <b>Vova Screener Bot</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ <b>Статус:</b> {status}\n"
        f"🕒 <b>Посл. скан:</b> {last_scan_time}\n"
        f"👥 <b>Юзеров:</b> 1 (Admin)\n" # Заглушка, можно расширить
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>Текущие настройки сканирования:</i>\n"
        f"SMA: {p['sma']} | TF: {p['tf']} | New: {p['new_only']}\n"
        f"Risk: {p['risk_pct']}% | RR: {p['min_rr']} | ATR: {p['max_atr']}%\n"
        f"Port: ${p['portfolio']:.0f}"
    )

async def check_auth(update: Update):
    user_id = update.effective_user.id
    # Простая проверка на админа для этого примера
    # В реальном коде можно делать запрос к GITHUB_USERS_URL
    if user_id != ADMIN_ID:
        await update.message.reply_text("⛔ Доступ запрещен.")
        return False
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update): return
    
    if 'params' not in context.user_data:
        context.user_data['params'] = DEFAULT_PARAMS.copy()
    
    context.user_data['scanning'] = False
    context.user_data['input_mode'] = None # Если ждем ввода числа
    
    p = context.user_data['params']
    await update.message.reply_html(
        get_status_text(p=p),
        reply_markup=get_keyboard(p)
    )

# --- SCAN ENGINE ---
async def run_scan_process(update, context, p, tickers, manual_input=False):
    """Основной процесс сканирования"""
    status_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="🚀 <b>Начинаю сканирование...</b>\n[░░░░░░░░░░] 0%", 
        parse_mode=constants.ParseMode.HTML
    )
    
    results_found = 0
    total = len(tickers)
    
    # SNAPSHOT PARAMETERS (Правило: старые параметры до конца скана)
    scan_p = p.copy()
    
    for i, t in enumerate(tickers):
        # Check Stop Flag
        if not context.user_data.get('scanning', False) and not manual_input:
            await status_msg.edit_text("⏹ Сканирование остановлено пользователем.")
            return

        # Progress Bar Logic (edit every 5% or 10 tickers to avoid flood limits)
        if i % 10 == 0 or i == total - 1:
            pct = int((i + 1) / total * 10)
            bar = "█" * pct + "░" * (10 - pct)
            try:
                await status_msg.edit_text(
                    f"🚀 <b>Сканирование:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n"
                    f"⚙️ <i>Исп. параметры: SMA{scan_p['sma']}, {scan_p['tf']}</i>"
                , parse_mode=constants.ParseMode.HTML)
            except: pass

        # --- DATA FETCH & LOGIC ---
        try:
            # Yield control to event loop
            await asyncio.sleep(0.05) 
            
            inter = "1d" if scan_p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if scan_p['tf'] == "Daily" else "5y"
            
            # Use threads=False to be safer inside async, though slightly slower
            df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            # A. Data Check
            if len(df) < scan_p['sma'] + 5:
                if manual_input:
                    await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b>: NO DATA", parse_mode='HTML')
                continue

            # B. Logic
            df = run_vova_logic(df, scan_p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            
            # C. Analyze
            valid, d, reason = analyze_trade(df, -1)
            
            if not valid:
                if manual_input:
                    pr = df['Close'].iloc[-1]
                    await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b> (${pr:.2f}): {reason}", parse_mode='HTML')
                continue

            # D. Filters
            # New Signal Check
            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if not manual_input and scan_p['new_only'] and not is_new: continue
            
            # Autoscan Unique Check
            if manual_input is False and scan_p.get('is_auto', False):
                if t in sent_today: continue
            
            # RR Check
            if d['RR'] < scan_p['min_rr']:
                if manual_input: await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b>: Low RR ({d['RR']:.2f})", parse_mode='HTML')
                continue
            
            # ATR Check
            atr_pct = (d['ATR']/d['P'])*100
            if atr_pct > scan_p['max_atr']:
                if manual_input: await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b>: High Vol ({atr_pct:.1f}%)", parse_mode='HTML')
                continue
            
            # E. Position Sizing
            risk_amt = scan_p['portfolio'] * (scan_p['risk_pct'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            
            shares = int(risk_amt / risk_share)
            max_shares_portfolio = int(scan_p['portfolio'] / d['P'])
            shares = min(shares, max_shares_portfolio)
            
            if shares < 1:
                if manual_input: await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b>: LOW FUNDS", parse_mode='HTML')
                continue
                
            # F. Prepare Output
            pe = get_financial_info(t)
            card_html = format_luxury_card(t, d, shares, is_new, pe)
            
            # Send Card
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=card_html,
                parse_mode=constants.ParseMode.HTML,
                disable_web_page_preview=True
            )
            
            if scan_p.get('is_auto', False):
                sent_today.add(t)
                
            results_found += 1
            
        except Exception as e:
            logger.error(f"Error processing {t}: {e}")
            if manual_input:
                 await context.bot.send_message(update.effective_chat.id, f"❌ <b>{t}</b>: Error {e}", parse_mode='HTML')

    global last_scan_time
    last_scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    end_msg = f"✅ <b>Скан завершен!</b> Найдено: {results_found}"
    await status_msg.edit_text(end_msg, parse_mode='HTML')
    context.user_data['scanning'] = False
    
    # Reshow menu
    await update.effective_message.reply_html(
         get_status_text("Готов", context.user_data['params']),
         reply_markup=get_keyboard(context.user_data['params'])
    )

# --- CALLBACKS & INTERACTION ---

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    p = context.user_data.get('params', DEFAULT_PARAMS)
    
    # Toggles
    if data == "toggle_tf":
        p['tf'] = "Weekly" if p['tf'] == "Daily" else "Daily"
    elif data == "toggle_new":
        p['new_only'] = not p['new_only']
    elif data == "toggle_auto":
        p['autoscan'] = not p['autoscan']
        if p['autoscan']:
            # Start job
            chat_id = update.effective_chat.id
            context.job_queue.run_repeating(auto_scan_job, interval=3600, first=10, chat_id=chat_id, user_id=ADMIN_ID, name=str(chat_id))
            await query.message.reply_text("🤖 Автоскан ВКЛЮЧЕН (Каждый час во время рынка US).")
        else:
            # Stop job
            current_jobs = context.job_queue.get_jobs_by_name(str(update.effective_chat.id))
            for job in current_jobs: job.schedule_removal()
            await query.message.reply_text("🤖 Автоскан ВЫКЛЮЧЕН.")
            
    elif data == "set_sma":
        opts = [100, 150, 200]
        try:
            curr_idx = opts.index(p['sma'])
            p['sma'] = opts[(curr_idx + 1) % len(opts)]
        except: p['sma'] = 200
        
    elif data == "start_scan":
        if context.user_data.get('scanning'):
            await query.message.reply_text("⚠️ Уже сканирую!")
            return
        
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        
        # Запуск в фоне, чтобы не блокировать
        asyncio.create_task(run_scan_process(update, context, p, tickers))
        return # Не обновляем меню сразу
        
    elif data == "stop_scan":
        context.user_data['scanning'] = False
        await query.message.reply_text("🛑 Остановка после текущего тикера...")
        return

    # Input Modes
    elif data in ["set_port", "set_rr", "set_risk", "set_matr"]:
        map_names = {
            "set_port": "Размер портфеля ($)",
            "set_rr": "Мин. Risk/Reward (>= 1.25)",
            "set_risk": "Риск на сделку % (>= 0.2)",
            "set_matr": "Макс ATR %"
        }
        context.user_data['input_mode'] = data
        await query.message.reply_text(f"✏️ Введите новое значение для: <b>{map_names[data]}</b>", parse_mode='HTML')
        return

    # Update Menu
    try:
        await query.message.edit_text(
            get_status_text("Настройка", p),
            reply_markup=get_keyboard(p),
            parse_mode=constants.ParseMode.HTML
        )
    except: pass

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update): return
    
    txt = update.message.text.strip()
    mode = context.user_data.get('input_mode')
    p = context.user_data.get('params', DEFAULT_PARAMS)
    
    # 1. Manual Ticker Scan (Comma separated)
    if not mode:
        if "," in txt or txt.isalpha() and len(txt) < 6:
            # Assume manual tickers
            raw_tickers = [x.strip().upper() for x in txt.split(",") if x.strip()]
            if raw_tickers:
                await update.message.reply_text(f"🔎 Ручной анализ: {', '.join(raw_tickers)}")
                # Run manual scan
                await run_scan_process(update, context, p, raw_tickers, manual_input=True)
            return
    
    # 2. Parameter Updates
    try:
        val = float(txt.replace(',', '.'))
        if mode == "set_port":
            p['portfolio'] = val
        elif mode == "set_rr":
            if val < 1.25: 
                await update.message.reply_text("⚠️ Минимум 1.25")
                return
            p['min_rr'] = val
        elif mode == "set_risk":
            if val < 0.2:
                await update.message.reply_text("⚠️ Минимум 0.2%")
                return
            p['risk_pct'] = val
        elif mode == "set_matr":
            p['max_atr'] = val
            
        context.user_data['input_mode'] = None
        await update.message.reply_html(
            f"✅ Значение принято.\n" + get_status_text("Готов", p),
            reply_markup=get_keyboard(p)
        )
        
    except ValueError:
        await update.message.reply_text("❌ Ошибка. Введите число.")

# --- JOB QUEUE (AUTOSCAN) ---

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    
    # 1. Reset 'sent_today' if new day
    global sent_today
    now_ny = datetime.datetime.now(pytz.timezone('US/Eastern'))
    if now_ny.hour == 9 and now_ny.minute < 5: # Reset around open
        sent_today.clear()
        
    # 2. Check Market Hours
    if not is_market_open():
        # Можно отправить лог админу, но лучше тихо пропустить
        return 

    # 3. Prepare Dummy Update object to reuse run_scan_process
    # We need to construct a fake object to pass chat_id
    class DummyChat:
        id = job.chat_id
    class DummyUpdate:
        effective_chat = DummyChat()
        effective_message = None # won't be used for reply in auto
        
    # Get user params from somewhere? 
    # JobQueue doesn't easily access user_data. 
    # For simplicity, we use DEFAULT or stored globally if single user.
    # Here we assume single admin use-case as requested.
    # Better way: pass params in job.data if needed, but here we just grab default or last known.
    
    # Note: Accessing context.application.user_data[user_id] is possible
    user_data = context.application.user_data.get(ADMIN_ID, {})
    p = user_data.get('params', DEFAULT_PARAMS)
    p['is_auto'] = True # Special flag
    
    await context.bot.send_message(job.chat_id, "🤖 <b>Автоскан запущен...</b>", parse_mode='HTML')
    
    tickers = get_sp500_tickers()
    await run_scan_process(DummyUpdate(), context, p, tickers)


# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================

if __name__ == '__main__':
    application = ApplicationBuilder().token(TG_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button_click))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_input))
    
    print("Bot is running...")
    application.run_polling(stop_signals=None)

