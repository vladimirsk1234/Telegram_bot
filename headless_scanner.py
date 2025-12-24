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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    PicklePersistence # <--- ДЛЯ СОХРАНЕНИЯ НАСТРОЕК
)

# --- КОНФИГУРАЦИЯ ---
nest_asyncio.apply()

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. ЗАГРУЗКА СЕКРЕТОВ
# ==========================================
try:
    TG_TOKEN = st.secrets["TG_TOKEN"]
    ADMIN_ID = int(st.secrets["ADMIN_ID"])
    GITHUB_USERS_URL = st.secrets.get("GITHUB_USERS_URL", "")
except Exception as e:
    st.error(f"❌ Ошибка секретов: {e}")
    st.stop()

# ==========================================
# 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ==========================================
# ACTIVE_USERS не нужен глобально для persistence, он будет в user_data
last_scan_time = "Никогда"
sent_today = set()

# Константы индикаторов
EMA_F = 20; EMA_S = 40; ADX_L = 14; ADX_T = 20; ATR_L = 14

# Дефолтные параметры
DEFAULT_PARAMS = {
    'portfolio': 10000.0,
    'min_rr': 1.25,
    'risk_pct': 0.2,
    'max_atr': 5.0,
    'sma': 200,
    'tf': 'Daily',
    'new_only': True,
    'autoscan': False,
}

# ==========================================
# 3. ЛОГИКА СКРИНЕРА (100% COPY)
# ==========================================
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: return []

def get_financial_info(ticker):
    try:
        t = yf.Ticker(ticker)
        i = t.info
        return i.get('trailingPE') or i.get('forwardPE')
    except: return None

# --- MATH ---
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

def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    ema_f = calc_ema(df['Close'], len_fast); ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close']); efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    n = len(df)
    c_a, h_a, l_a = df['Close'].values, df['High'].values, df['Low'].values
    seq_st = np.zeros(n, dtype=int); crit_lvl = np.full(n, np.nan)
    res_peak = np.full(n, np.nan); res_struct = np.zeros(n, dtype=bool)
    
    s_state = 0; s_crit = np.nan; s_h = h_a[0]; s_l = l_a[0]
    last_pk = np.nan; last_tr = np.nan; pk_hh = False; tr_hl = False
    
    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        prev_st = s_state; prev_cr = s_crit; prev_sh = s_h; prev_sl = s_l
        brk = False
        if prev_st == 1 and not np.isnan(prev_cr): brk = c < prev_cr
        elif prev_st == -1 and not np.isnan(prev_cr): brk = c > prev_cr
        if brk:
            if prev_st == 1:
                is_hh = True if np.isnan(last_pk) else (prev_sh > last_pk)
                pk_hh = is_hh; last_pk = prev_sh; s_state = -1; s_h = h; s_l = l; s_crit = h
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
                if c > prev_sh: s_state = 1; s_crit = l
                elif c < prev_sl: s_state = -1; s_crit = h
                else: s_h = max(prev_sh, h); s_l = min(prev_sl, l)
        seq_st[i] = s_state; crit_lvl[i] = s_crit; res_peak[i] = last_pk; res_struct[i] = (pk_hh and tr_hl)

    adx_str = adx >= adx_thr
    bull = (adx_str & (p_di > m_di)) & ((ema_f > ema_f.shift(1)) & (ema_s > ema_s.shift(1)) & (hist > hist.shift(1))) & (efi > 0)
    bear = (adx_str & (m_di > p_di)) & ((ema_f < ema_f.shift(1)) & (ema_s < ema_s.shift(1)) & (hist < hist.shift(1))) & (efi < 0)
    t_st = np.zeros(n, dtype=int); t_st[bull] = 1; t_st[bear] = -1
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
    
    price = r['Close']; tp = r['Peak']; crit = r['Crit']; atr = r['ATR']
    final_sl = min(crit, price - atr)
    risk = price - final_sl; reward = tp - price
    if risk <= 0: return False, {}, "BAD STOP"
    if reward <= 0: return False, {}, "AT TARGET"
    
    return True, {
        "P": price, "TP": tp, "SL": final_sl, 
        "RR": reward/risk, "ATR": atr, "Crit": crit,
        "SL_Type": "STR" if abs(final_sl - crit) < 0.01 else "ATR"
    }, "OK"

# ==========================================
# 4. HELPER FUNCTIONS
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
    # Сохраняем активных в user_data чтобы они сохранялись в pickle
    if 'active_users' not in context.bot_data: context.bot_data['active_users'] = set()
    context.bot_data['active_users'].add(user_id)
    
    allowed = get_allowed_users()
    if user_id not in allowed:
        await update.message.reply_text("⛔ Доступ запрещен.", parse_mode='HTML')
        return False
    return True

def format_luxury_card(ticker, d, shares, is_new, pe_val):
    tv_link = f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"
    badge = "🆕" if is_new else ""
    pe_str = f"| P/E: <b>{pe_val:.0f}</b>" if pe_val else ""
    val_pos = shares * d['P']
    profit = (d['TP'] - d['P']) * shares
    loss = (d['P'] - d['SL']) * shares
    atr_pct = (d['ATR'] / d['P']) * 100
    
    return (
        f"💎 <b><a href='{tv_link}'>{ticker}</a></b> {badge}\n"
        f"💵 <b>{d['P']:.2f}</b> {pe_str}\n"
        f"💼 <b>POS:</b> {shares} (<b>${val_pos:.0f}</b>) | ⚖️ <b>R:R:</b> {d['RR']:.2f}\n"
        f"🎯 <b>TP:</b> {d['TP']:.2f} (<span class='tg-spoiler'>+${profit:.0f}</span>)\n"
        f"🛑 <b>SL:</b> {d['SL']:.2f} (<span class='tg-spoiler'>-${loss:.0f}</span>) [{d['SL_Type']}]\n"
        f"📉 <b>Crit:</b> {d['Crit']:.2f}\n"
        f"📊 <b>ATR:</b> {d['ATR']:.2f} ({atr_pct:.1f}%)"
    )

def get_keyboard(p):
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
            InlineKeyboardButton(f"AutoScan: {auto_txt}", callback_data="toggle_auto"),
        ],
        [
            InlineKeyboardButton("▶️ START SCAN", callback_data="start_scan"),
            InlineKeyboardButton("⏹ STOP", callback_data="stop_scan"),
        ]
    ]
    return InlineKeyboardMarkup(kb)

def get_status_text(status="💤 Ожидание", p=None):
    if not p: return f"Статус: {status}"
    return (
        f"🖥 <b>Vova Screener Bot</b>\n━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ <b>Статус:</b> {status}\n"
        f"🕒 <b>Посл. скан:</b> {last_scan_time}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"SMA: {p['sma']} | TF: {p['tf']} | New: {p['new_only']}\n"
        f"Risk: {p['risk_pct']}% | RR: {p['min_rr']} | ATR: {p['max_atr']}%\n"
        f"Port: ${p['portfolio']:.0f}"
    )

# ==========================================
# 5. MENUS & UTILS (STICKY BOTTOM)
# ==========================================

async def refresh_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, p, status="Готов"):
    """
    Удаляет старое сообщение и присылает новое внизу.
    """
    try:
        # Пытаемся удалить старое сообщение (если это callback или просто текст)
        if update.callback_query:
            await update.callback_query.message.delete()
        elif update.message:
            # Не удаляем сообщение пользователя, чтобы он видел что ввел, 
            # но можно удалить предыдущее сообщение бота если сохранить ID.
            # Для простоты просто шлем новое.
            pass
    except: pass
    
    # Шлем новое
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=get_status_text(status, p),
        reply_markup=get_keyboard(p),
        parse_mode='HTML'
    )
    # Можно сохранить msg.message_id в user_data чтобы потом его удалять, 
    # но это усложнит логику. Пока просто "новое сообщение всегда внизу".

async def safe_get_params(context):
    """Гарантирует, что параметры не сбросятся"""
    if 'params' not in context.user_data:
        context.user_data['params'] = DEFAULT_PARAMS.copy()
    return context.user_data['params']

# ==========================================
# 6. SCAN PROCESS
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_input=False):
    # Присылаем сообщение статуса (оно будет висеть внизу во время скана)
    status_msg = await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="🚀 <b>Начинаю сканирование...</b>", 
        parse_mode=constants.ParseMode.HTML
    )
    
    results_found = 0
    total = len(tickers)
    scan_p = p.copy() 
    
    for i, t in enumerate(tickers):
        if not context.user_data.get('scanning', False) and not manual_input:
            await status_msg.edit_text("⏹ Сканирование остановлено.")
            break

        if i % 10 == 0 or i == total - 1:
            pct = int((i + 1) / total * 10)
            bar = "█" * pct + "░" * (10 - pct)
            try:
                await status_msg.edit_text(
                    f"🚀 <b>Scan:</b> {i+1}/{total}\n[{bar}] {int((i+1)/total*100)}%\n"
                    f"<i>Params: SMA{scan_p['sma']}, {scan_p['tf']}</i>", 
                    parse_mode='HTML'
                )
            except: pass

        try:
            await asyncio.sleep(0.01)
            inter = "1d" if scan_p['tf'] == "Daily" else "1wk"
            fetch_period = "2y" if scan_p['tf'] == "Daily" else "5y"
            
            df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
            
            if len(df) < scan_p['sma'] + 5:
                if manual_input: await context.bot.send_message(update.effective_chat.id, f"❌ {t}: NO DATA")
                continue

            df = run_vova_logic(df, scan_p['sma'], EMA_F, EMA_S, ADX_L, ADX_T, ATR_L)
            valid, d, reason = analyze_trade(df, -1)
            
            if not valid:
                if manual_input: await context.bot.send_message(update.effective_chat.id, f"❌ {t}: {reason}")
                continue

            valid_prev, _, _ = analyze_trade(df, -2)
            is_new = not valid_prev
            
            if not manual_input and scan_p['new_only'] and not is_new: continue
            if not manual_input and scan_p.get('is_auto') and t in sent_today: continue
            if d['RR'] < scan_p['min_rr']: continue
            if (d['ATR']/d['P'])*100 > scan_p['max_atr']: continue
            
            risk_amt = scan_p['portfolio'] * (scan_p['risk_pct'] / 100.0)
            risk_share = d['P'] - d['SL']
            if risk_share <= 0: continue
            shares = min(int(risk_amt / risk_share), int(scan_p['portfolio'] / d['P']))
            if shares < 1: continue
            
            pe = get_financial_info(t)
            card = format_luxury_card(t, d, shares, is_new, pe)
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=card,
                parse_mode=constants.ParseMode.HTML,
                disable_web_page_preview=True
            )
            
            if scan_p.get('is_auto'): sent_today.add(t)
            results_found += 1
            
        except: pass

    global last_scan_time
    last_scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Удаляем сообщение прогресса
    try: await status_msg.delete()
    except: pass
    
    # Итоговый отчет
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"✅ <b>Скан завершен!</b> Найдено: {results_found}",
        parse_mode='HTML'
    )
    context.user_data['scanning'] = False
    
    # ВОЗВРАЩАЕМ МЕНЮ ВНИЗ
    if not manual_input and not scan_p.get('is_auto'):
        await refresh_menu(update, context, p)

# ==========================================
# 7. HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    p = await safe_get_params(context)
    context.user_data['scanning'] = False
    context.user_data['input_mode'] = None
    
    # Первое меню
    await update.message.reply_html(
        get_status_text(p=p),
        reply_markup=get_keyboard(p)
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    active = context.bot_data.get('active_users', set())
    allowed = get_allowed_users()
    msg = (
        f"📊 <b>АДМИН</b>\n"
        f"Активных: {len(active)}\n"
        f"Whitelist: {len(allowed)}\n"
        f"Скан: {last_scan_time}\n"
        f"IDs: {list(active)}"
    )
    await update.message.reply_html(msg)

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    p = await safe_get_params(context)
    
    # Обработка кнопок
    if data == "toggle_tf": p['tf'] = "Weekly" if p['tf'] == "Daily" else "Daily"
    elif data == "toggle_new": p['new_only'] = not p['new_only']
    elif data == "toggle_auto":
        p['autoscan'] = not p['autoscan']
        if p['autoscan']:
            chat_id = update.effective_chat.id
            context.job_queue.run_repeating(auto_scan_job, interval=3600, first=10, chat_id=chat_id, user_id=ADMIN_ID, name=str(chat_id))
            await context.bot.send_message(chat_id, "🤖 Автоскан ВКЛ.")
        else:
            for job in context.job_queue.get_jobs_by_name(str(update.effective_chat.id)): job.schedule_removal()
            await context.bot.send_message(update.effective_chat.id, "🤖 Автоскан ВЫКЛ.")
            
    elif data == "set_sma":
        opts = [100, 150, 200]
        try: p['sma'] = opts[(opts.index(p['sma']) + 1) % 3]
        except: p['sma'] = 200
        
    elif data == "start_scan":
        if context.user_data.get('scanning'):
            await context.bot.send_message(update.effective_chat.id, "⚠️ Уже сканирую!")
            return
        context.user_data['scanning'] = True
        
        # Удаляем старое меню перед сканом
        try: await query.message.delete()
        except: pass
        
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers))
        return # Выход, меню будет отправлено в конце скана

    elif data == "stop_scan":
        context.user_data['scanning'] = False
        await context.bot.send_message(update.effective_chat.id, "🛑 Остановка...")
        return

    elif data in ["set_port", "set_rr", "set_risk", "set_matr"]:
        context.user_data['input_mode'] = data
        try: await query.message.delete()
        except: pass
        await context.bot.send_message(update.effective_chat.id, f"✏️ Введите значение:", parse_mode='HTML')
        return

    # ОБНОВЛЕНИЕ МЕНЮ (Удалить старое -> Прислать новое)
    await refresh_menu(update, context, p)

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    p = await safe_get_params(context)
    txt = update.message.text.strip()
    mode = context.user_data.get('input_mode')
    
    if not mode:
        if "," in txt or (txt.isalpha() and len(txt)<6):
            ts = [x.strip().upper() for x in txt.split(",") if x.strip()]
            if ts:
                await update.message.reply_text(f"🔎 Manual: {ts}")
                await run_scan_process(update, context, p, ts, manual_input=True)
            return
    
    try:
        val = float(txt.replace(',', '.'))
        if mode == "set_port": p['portfolio'] = val
        elif mode == "set_rr": p['min_rr'] = max(1.25, val)
        elif mode == "set_risk": p['risk_pct'] = max(0.2, val)
        elif mode == "set_matr": p['max_atr'] = val
        context.user_data['input_mode'] = None
        
        # Шлем новое меню вниз
        await refresh_menu(update, context, p, status="Параметр обновлен")
        
    except: await update.message.reply_text("❌ Введите число.")

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    global sent_today
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    if now.hour == 9 and now.minute < 5: sent_today.clear()
    if not is_market_open(): return 
    
    class Dummy: pass
    u = Dummy(); u.effective_chat = Dummy(); u.effective_chat.id = job.chat_id
    
    # Пытаемся восстановить параметры из persistence
    if 'params' not in context.application.user_data.get(job.user_id, {}):
         context.application.user_data.setdefault(job.user_id, {})['params'] = DEFAULT_PARAMS.copy()
    
    p = context.application.user_data[job.user_id]['params'].copy()
    p['is_auto'] = True
    
    await context.bot.send_message(job.chat_id, "🤖 <b>Автоскан...</b>", parse_mode='HTML')
    await run_scan_process(u, context, p, get_sp500_tickers())

# ==========================================
# 8. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    st.title("💎 Vova Screener Bot Running")
    st.write("Бот запущен с защитой настроек.")
    
    # Persistence сохраняет данные в файл 'bot_data.pickle'
    my_persistence = PicklePersistence(filepath='bot_data.pickle')
    
    application = ApplicationBuilder().token(TG_TOKEN).persistence(my_persistence).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('stats', stats_command))
    application.add_handler(CallbackQueryHandler(button_click))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_input))
    
    print("Bot started...")
    try:
        application.run_polling(stop_signals=None, close_loop=False)
    except Exception as e:
        st.error(f"Critical Error: {e}")
