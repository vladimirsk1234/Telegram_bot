import telebot
import yfinance as yf
import pandas as pd
import numpy as np
import io
import time
import threading
import requests

# ==========================================
# 1. НАСТРОЙКИ (Ваши данные)
# ==========================================
TG_TOKEN = "8407386703:AAEFkQ66ZOcGd7Ru41hrX34Bcb5BriNPuuQ"
# Chat ID бот запомнит сам после /start

# Инициализация бота
bot = telebot.TeleBot(TG_TOKEN)

# Глобальные настройки
SETTINGS = {
    "LENGTH_MAJOR": 200,
    "MAX_ATR_PCT": 5.0,      # Максимальная волатильность 5%
    "ADX_THRESH": 20,        # Внутренний порог ADX (скрыт из меню)
    "AUTO_SCAN_INTERVAL": 60,
    "IS_SCANNING": False,
    "STOP_SCAN": False,
    "SHOW_ONLY_NEW": True    # True = Только новые входы, False = Все зеленые
}

# ==========================================
# 2. ФУНКЦИИ АНАЛИЗА
# ==========================================
def get_sp500_tickers():
    # Пробуем скачать список 3 раза, если ошибка сети
    for attempt in range(3):
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            table = pd.read_html(io.StringIO(response.text))
            tickers = table[0]['Symbol'].tolist()
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                print(f"Не удалось получить S&P 500: {e}")
                return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "JPM", "BAC"]

def pine_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def check_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if len(df) < 250: return None

        # SMA
        df['SMA_Major'] = df['Close'].rolling(window=SETTINGS["LENGTH_MAJOR"]).mean()
        
        # ATR Calculation
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR_Val'] = df['TR'].rolling(window=14).mean()
        df['ATR_Pct'] = (df['ATR_Val'] / df['Close']) * 100
        
        # ADX Logic
        df['Up'] = df['High'] - df['High'].shift(1)
        df['Down'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['Up'] > df['Down']) & (df['Up'] > 0), df['Up'], 0)
        df['-DM'] = np.where((df['Down'] > df['Up']) & (df['Down'] > 0), df['Down'], 0)
        tr_smooth = pine_rma(df['TR'], 14)
        plus_dm = pine_rma(df['+DM'], 14)
        minus_dm = pine_rma(df['-DM'], 14)
        df['DI_Plus'] = 100 * (plus_dm / tr_smooth)
        df['DI_Minus'] = 100 * (minus_dm / tr_smooth)
        dx = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = pine_rma(dx, 14)

        # Sequence Logic
        seqState = 0; seqHigh = df['High'].iloc[0]; seqLow = df['Low'].iloc[0]; criticalLevel = df['Low'].iloc[0]
        df_calc = df.iloc[-300:].copy()
        closes = df_calc['Close'].values; highs = df_calc['High'].values; lows = df_calc['Low'].values
        seq_states = []
        
        for i in range(len(df_calc)):
            c, h, l = closes[i], highs[i], lows[i]
            if i == 0: seq_states.append(0); continue
            
            pS = seq_states[-1]
            brk = (pS == 1 and c < criticalLevel) or (pS == -1 and c > criticalLevel)
            
            if brk:
                if pS == 1: seqState = -1; seqHigh = h; seqLow = l; criticalLevel = h
                else: seqState = 1; seqHigh = h; seqLow = l; criticalLevel = l
            else:
                if seqState == 1:
                    if h >= seqHigh: seqHigh = h
                    criticalLevel = l if h >= seqHigh else criticalLevel
                elif seqState == -1:
                    if l <= seqLow: seqLow = l
                    criticalLevel = h if l <= seqLow else criticalLevel
                else:
                    if c > seqHigh: seqState = 1; criticalLevel = l
                    elif c < seqLow: seqState = -1; criticalLevel = h
                    else: seqHigh = max(seqHigh, h); seqLow = min(seqLow, l)
            seq_states.append(seqState)

        # CHECK LAST BAR
        last = df_calc.iloc[-1]
        prev = df_calc.iloc[-2]
        
        if pd.isna(last['ADX']): return None
        
        # Current Status
        seq_cur = seq_states[-1] == 1
        ma_cur = last['Close'] > last['SMA_Major']
        # Используем ADX в логике, но не показываем в настройках
        mom_cur = (last['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_cur and (last['DI_Plus'] > last['DI_Minus'])
        all_green_cur = seq_cur and ma_cur and mom_cur
        
        # Previous Status (для определения новизны)
        seq_prev = seq_states[-2] == 1
        ma_prev = prev['Close'] > prev['SMA_Major']
        mom_prev = (prev['ADX'] >= SETTINGS["ADX_THRESH"]) and seq_prev and (prev['DI_Plus'] > prev['DI_Minus'])
        all_green_prev = seq_prev and ma_prev and mom_prev
        
        # Filter: ATR Check
        pass_filters = (last['ATR_Pct'] <= SETTINGS["MAX_ATR_PCT"])
        
        # Is New Signal?
        is_new_signal = all_green_cur and not all_green_prev

        # Logic: Return if we pass filters AND (we want all greens OR (we want only new AND it is new))
        if all_green_cur and pass_filters:
            if not SETTINGS["SHOW_ONLY_NEW"] or is_new_signal:
                return {
                    'ticker': ticker,
                    'price': last['Close'],
                    'atr': last['ATR_Pct'],
                    'is_new': is_new_signal
                }
    except: return None
    return None

def perform_scan(chat_id):
    if SETTINGS["IS_SCANNING"]:
        try:
            bot.send_message(chat_id, "⚠️ Сканирование уже идет! Введите /stop для отмены.")
        except: pass
        return

    SETTINGS["IS_SCANNING"] = True
    SETTINGS["STOP_SCAN"] = False
    
    mode_text = "ТОЛЬКО НОВЫЕ (вход сегодня)" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ ЗЕЛЕНЫЕ (текущий тренд)"
    try:
        bot.send_message(chat_id, f"🚀 <b>Начинаю сканирование S&P 500...</b>\nРежим: {mode_text}\nMax ATR: {SETTINGS['MAX_ATR_PCT']}%\nПодождите 1-2 минуты.", parse_mode="HTML")
    except: pass
    
    tickers = get_sp500_tickers()
    found_count = 0
    
    for i, t in enumerate(tickers):
        if SETTINGS["STOP_SCAN"]:
            try:
                bot.send_message(chat_id, "🛑 Сканирование остановлено.")
            except: pass
            SETTINGS["IS_SCANNING"] = False
            return

        res = check_ticker(t)
        if res:
            found_count += 1
            icon = "🔥 NEW" if res['is_new'] else "🟢"
            msg = f"{icon} <b>{res['ticker']}</b> | ${res['price']:.2f} | ATR: {res['atr']:.2f}%"
            try:
                bot.send_message(chat_id, msg, parse_mode="HTML")
            except Exception as e:
                print(f"Ошибка отправки сообщения: {e}")
    
    try:
        if found_count == 0:
            bot.send_message(chat_id, "🤷‍♂️ Ничего не найдено с текущими фильтрами.")
        else:
            bot.send_message(chat_id, f"✅ Готово. Найдено: {found_count}")
    except: pass
    
    SETTINGS["IS_SCANNING"] = False

# ==========================================
# 3. TELEGRAM КОМАНДЫ
# ==========================================

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 
        "👋 <b>Привет! Я Vova S&P 500 Screener Bot.</b>\n\n"
        "Я сканирую акции из индекса S&P 500 и ищу те, у которых загорелись <b>3 зеленых сигнала</b>:\n"
        "1. 🟢 <b>Price > SMA</b> (Глобальный тренд)\n"
        "2. 🟢 <b>Sequence</b> (Бычья структура HH/HL)\n"
        "3. 🟢 <b>Trend</b> (ADX > 20 + DI+ > DI-)\n\n"
        "<b>⚙️ Доступные параметры и команды:</b>\n\n"
        "🔍 <b>Управление поиском:</b>\n"
        "/scan - 🚀 Запустить сканирование вручную\n"
        "/stop - 🛑 Остановить текущее сканирование\n"
        "/mode - 🔄 Переключить режим (Искать только новые входы сегодня или все активные тренды)\n"
        "/status - 📊 Показать текущие настройки\n\n"
        "🛠 <b>Настройка фильтров:</b>\n"
        "/set_atr 5.0 - Установить Макс. волатильность (ATR %). Отсеивает слишком резкие акции.\n"
        "/set_sma 200 - Установить период скользящей средней (SMA).\n\n"
        "<i>Нажмите на команду, чтобы скопировать или запустить.</i>",
        parse_mode="HTML"
    )

@bot.message_handler(commands=['scan'])
def command_scan(message):
    threading.Thread(target=perform_scan, args=(message.chat.id,)).start()

@bot.message_handler(commands=['stop'])
def command_stop(message):
    if SETTINGS["IS_SCANNING"]:
        SETTINGS["STOP_SCAN"] = True
        bot.reply_to(message, "🛑 Останавливаю...")
    else:
        bot.reply_to(message, "⚠️ Сейчас ничего не сканируется.")

@bot.message_handler(commands=['mode'])
def command_mode(message):
    SETTINGS["SHOW_ONLY_NEW"] = not SETTINGS["SHOW_ONLY_NEW"]
    state = "ТОЛЬКО НОВЫЕ СИГНАЛЫ (Вход сегодня)" if SETTINGS["SHOW_ONLY_NEW"] else "ВСЕ ЗЕЛЕНЫЕ (Любой активный тренд)"
    bot.reply_to(message, f"🔄 Режим изменен:\n<b>{state}</b>", parse_mode="HTML")

@bot.message_handler(commands=['status'])
def command_status(message):
    mode_str = "Только Новые" if SETTINGS["SHOW_ONLY_NEW"] else "Все Зеленые"
    msg = (
        f"⚙️ <b>Текущие настройки:</b>\n"
        f"• Режим: <b>{mode_str}</b>\n"
        f"• SMA Period: {SETTINGS['LENGTH_MAJOR']}\n"
        f"• Max ATR: {SETTINGS['MAX_ATR_PCT']}%\n"
        f"(ADX используется скрыто > 20)"
    )
    bot.send_message(message.chat.id, msg, parse_mode="HTML")

@bot.message_handler(commands=['set_atr'])
def set_atr(message):
    try:
        val = float(message.text.split()[1])
        SETTINGS["MAX_ATR_PCT"] = val
        bot.reply_to(message, f"✅ Max ATR установлен на {val}%")
    except:
        bot.reply_to(message, "❌ Пример: /set_atr 5.0")

@bot.message_handler(commands=['set_sma'])
def set_sma(message):
    try:
        val = int(message.text.split()[1])
        SETTINGS["LENGTH_MAJOR"] = val
        bot.reply_to(message, f"✅ SMA Period установлен на {val}")
    except:
        bot.reply_to(message, "❌ Пример: /set_sma 200")

# ==========================================
# 4. ЗАПУСК БОТА (С ЗАЩИТОЙ ОТ ПАДЕНИЙ)
# ==========================================
if __name__ == "__main__":
    print("🤖 Бот запущен! Пишите /scan в Telegram.")
    while True:
        try:
            # infinity_polling автоматически перезапускает бота при разрывах связи
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            print(f"⚠️ Ошибка соединения: {e}")
            time.sleep(5) # Ждем 5 сек перед перезапуском