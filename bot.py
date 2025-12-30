import logging
import asyncio
import datetime
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import nest_asyncio
import time
import os
import gc
import warnings

from numba import jit
import numpy as np

# ==========================================
# NUMBA ENGINE (Speed x500)
# ==========================================
@jit(nopython=True, cache=False)
def calculate_structure_engine(c_a, h_a, l_a):
    n = len(c_a)
    seq_st = np.zeros(n, dtype=np.int64)
    crit_lvl = np.full(n, np.nan, dtype=np.float64)
    res_peak = np.full(n, np.nan, dtype=np.float64)
    res_struct = np.zeros(n, dtype=np.bool_)
    
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
        
        if pk_hh and tr_hl:
            res_struct[i] = True
        else:
            res_struct[i] = False
            
    return seq_st, crit_lvl, res_peak, res_struct

warnings.simplefilter(action='ignore', category=FutureWarning)

from telegram import (
    Update, 
    ReplyKeyboardMarkup, 
    KeyboardButton, 
    constants,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    PicklePersistence,
    Application,
    ChatJoinRequestHandler,
    CallbackQueryHandler
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
# LOAD FROM ENVIRONMENT VARIABLES (Railway)
# ==========================================
TG_TOKEN = os.environ.get("TG_TOKEN", "").strip()
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))
GITHUB_USERS_URL = os.environ.get("GITHUB_USERS_URL", "").strip()
CHANNEL_ID = os.environ.get("CHANNEL_ID", None)
if CHANNEL_ID: 
    CHANNEL_ID = int(CHANNEL_ID)

if not TG_TOKEN:
    print("❌ ERROR: TG_TOKEN environment variable not set!")
    exit(1)
if not ADMIN_ID:
    print("❌ ERROR: ADMIN_ID environment variable not set!")
    exit(1)

print(f"✅ Config loaded: Token={TG_TOKEN[:10]}... | Admin={ADMIN_ID} | Channel={CHANNEL_ID}")

# GLOBAL SETTINGS
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
# CORE LOGIC
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        html = pd.read_html(requests.get(url, headers=headers).text, header=0)
        return [t.replace('.', '-') for t in html[0]['Symbol'].tolist()]
    except: 
        return []

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

def run_vova_logic(df, len_maj, len_fast, len_slow, adx_len, adx_thr, atr_len):
    df['SMA'] = calc_sma(df['Close'], len_maj)
    adx, p_di, m_di = calc_adx_pine(df, adx_len)
    
    ema_f = calc_ema(df['Close'], len_fast)
    ema_s = calc_ema(df['Close'], len_slow)
    hist = calc_macd(df['Close'])
    efi = calc_ema(df['Close'].diff() * df['Volume'], len_fast)
    atr = calc_atr(df, atr_len)
    
    c_vals = df['Close'].values.astype(np.float64)
    h_vals = df['High'].values.astype(np.float64)
    l_vals = df['Low'].values.astype(np.float64)
    
    seq_st, crit_lvl, res_peak, res_struct = calculate_structure_engine(c_vals, h_vals, l_vals)
    
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
# UI FORMATTING
# ==========================================
def format_dashboard_card(ticker, d, shares, is_new, info, p_risk, sma_len, public_view=False):
    tv_ticker = ticker.replace('-', '.')
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_ticker}"
    
    pe_str = str(info.get('pe', 'N/A'))
    mc_str = str(info.get('mc', 'N/A'))
    atr_pct = (d['ATR'] / d['Close']) * 100
    
    trend_emo = "🟢" if d['Trend'] == 1 else ("🔴" if d['Trend'] == -1 else "🟡")
    seq_emo = "🟢" if d['Seq'] == 1 else ("🔴" if d['Seq'] == -1 else "🟡")
    ma_emo = "🟢" if d['Close'] > d['SMA'] else "🔴"
    status_icon = "🆕" if is_new else "♻️"

    header = f"<b><a href='{tv_link}'>{ticker}</a></b>  ${d['P']:.2f}\n"
    context_block = (
        f"MC: {mc_str} | P/E: {pe_str}\n"
        f"ATR: ${d['ATR']:.2f} ({atr_pct:.2f}%)\n"
        f"Trend {trend_emo}  Seq {seq_emo}  MA{sma_len} {ma_emo}\n"
    )

    cond_seq = d['Seq'] == 1
    cond_ma = d['Close'] > d['SMA']
    cond_trend = d['Trend'] != -1
    cond_struct = d.get('Struct', False)
    is_valid_setup = cond_seq and cond_ma and cond_trend and cond_struct
    risk = d['P'] - d['SL']
    reward = d['TP'] - d['P']
    is_valid_math = risk > 0 and reward > 0

    if is_valid_setup and is_valid_math:
        rr_str = f"{d['RR']:.2f}"
        
        if public_view:
            html = (
                f"{status_icon} {header}"
                f"{context_block}"
                f"🛑 SL: {d['SL']:.2f}\n"
                f"🎯 TP: {d['TP']:.2f}\n"
                f"⚖️ Risk/Reward: {rr_str}"
            )
            return html
        else:
            profit = reward * shares
            loss = risk * shares
            total_val = shares * d['P']
            
            size_line = f"Size: <b>{shares}</b> shares (${total_val:,.0f})\n"
            sl_line = f"🛑 SL: {d['SL']:.2f}  (-${loss:.0f})\n"
            tp_line = f"🎯 TP: {d['TP']:.2f}  (+${profit:.0f})\n"

            html = (
                f"{status_icon} {header}"
                f"{size_line}"
                f"{context_block}"
                f"{sl_line}"
                f"{tp_line}"
                f"⚖️ Risk/Reward: {rr_str}"
            )
            return html
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
# SCANNING PROCESS
# ==========================================
async def run_scan_process(update, context, p, tickers, manual_mode=False, is_auto=False):
    if update.effective_chat:
        target_chat_id = update.effective_chat.id
    else:
        target_chat_id = ADMIN_ID
    
    ny_tz = pytz.timezone('US/Eastern')
    today_str = datetime.datetime.now(ny_tz).strftime('%Y-%m-%d')
    
    if 'channel_mem' not in context.bot_data: 
        context.bot_data['channel_mem'] = {'date': today_str, 'tickers': []}
    if context.bot_data['channel_mem']['date'] != today_str:
        context.bot_data['channel_mem'] = {'date': today_str, 'tickers': []}

    config_display = (
        f"⚙️ <b>Active Settings:</b>\n"
        f"💰 Risk: <b>${p['risk_usd']:.0f}</b>\n"
        f"⚖️ Min RR: <b>{p['min_rr']}</b>\n"
        f"📊 Max ATR: <b>{p['max_atr']}%</b>\n"
        f"📈 SMA Filter: <b>{p['sma']}</b>\n"
        f"⏳ Timeframe: <b>{p['tf']}</b>\n"
        f"🆕 Fresh Only: <b>{'✅' if p['new_only'] else '❌'}</b>"
    )
    
    scan_type = "🤖 AUTO" if is_auto else "👤 MANUAL"
    
    try:
        status_msg = await context.bot.send_message(
            chat_id=target_chat_id, 
            text=f"🚀 <b>{scan_type} Scan Started...</b>\n{config_display}", 
            parse_mode='HTML'
        )
    except:
        status_msg = None
    
    results_found = 0
    total = len(tickers)
    
    inter = "1d" if p['tf'] == "Daily" else "1wk"
    fetch_period = "2y" if p['tf'] == "Daily" else "5y"
    
    try:
        if status_msg:
            await status_msg.edit_text(
                f"📥 <b>Downloading data for {total} tickers...</b>\n\n{config_display}",
                parse_mode='HTML'
            )
    except: pass
    
    all_data = None
    try:
        all_data = yf.download(
            tickers, 
            period=fetch_period, 
            interval=inter, 
            progress=False, 
            auto_adjust=False, 
            group_by='ticker',
            threads=True
        )
        if all_data is None or all_data.empty:
            raise ValueError("Batch download returned empty data")
    except Exception as e:
        logger.warning(f"Batch download failed: {e}. Falling back to sequential downloads.")
        all_data = None
    
    try:
        if status_msg:
            await status_msg.edit_text(
                f"🔎 <b>{scan_type} Processing {total} tickers...</b>\n\n{config_display}",
                parse_mode='HTML'
            )
    except: pass
    
    for i, t in enumerate(tickers):
        if not is_auto and not context.user_data.get('scanning', False):
            await context.bot.send_message(target_chat_id, "⏹ <b>Scan Stopped.</b>", parse_mode='HTML')
            break
            
        if i % 10 == 0 or i == total - 1:
            try:
                pct = int((i + 1) / total * 10)
                bar = "█" * pct + "░" * (10 - pct)
                percent_num = int((i + 1) / total * 100)
                
                if status_msg:
                    await status_msg.edit_text(
                        f"🔎 <b>{scan_type} Scanning...</b>\n"
                        f"[{bar}] {percent_num}%\n"
                        f"👉 Checking: <b>{t}</b> ({i+1}/{total})\n\n"
                        f"{config_display}",
                        parse_mode='HTML'
                    )
            except: pass
            
        if i % 50 == 0: 
            gc.collect()
            await asyncio.sleep(0.05)
        
        try:
            await asyncio.sleep(0.02)
            
            df = None
            if all_data is not None and not all_data.empty:
                if isinstance(all_data.columns, pd.MultiIndex):
                    ticker_level = all_data.columns.get_level_values(0).unique()
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    has_all_cols = all((t, col) in all_data.columns for col in required_cols)
                    
                    if t in ticker_level and has_all_cols:
                        try:
                            df = pd.DataFrame({
                                'Open': all_data[(t, 'Open')],
                                'High': all_data[(t, 'High')],
                                'Low': all_data[(t, 'Low')],
                                'Close': all_data[(t, 'Close')],
                                'Volume': all_data[(t, 'Volume')]
                            })
                            if df.empty or df['Close'].isna().all():
                                raise ValueError("Empty or invalid data")
                        except (KeyError, ValueError):
                            df = None
                            try:
                                df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if df is None or df.empty or not all(col in df.columns for col in required_cols):
                                    df = None
                            except Exception:
                                df = None
                else:
                    if len(tickers) == 1 and t == tickers[0]:
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in all_data.columns for col in required_cols):
                            df = all_data[required_cols].copy()
                        else:
                            df = None
                    else:
                        df = None
            else:
                try:
                    df = yf.download(t, period=fetch_period, interval=inter, progress=False, auto_adjust=False, multi_level_index=False)
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_cols):
                        df = None
                except Exception:
                    df = None
            
            if df is None or df.empty or len(df) < p['sma'] + 5:
                continue
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                continue
            
            df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
            if len(df) < p['sma'] + 5:
                continue
            
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
                await asyncio.sleep(0.1)
                risk_per_share = d['P'] - d['SL']
                shares = int(p['risk_usd'] / risk_per_share) if risk_per_share > 0 else 0
                
                if is_auto and CHANNEL_ID:
                    if t not in context.bot_data['channel_mem']['tickers']:
                        public_card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'], p['sma'], public_view=True)
                        final_msg = public_card
                        
                        try:
                            await context.bot.send_message(chat_id=CHANNEL_ID, text=final_msg, parse_mode='HTML', disable_web_page_preview=True)
                            context.bot_data['channel_mem']['tickers'].append(t)
                            results_found += 1
                            await asyncio.sleep(0.2)
                        except telegram.error.TimedOut:
                            logger.warning(f"Timeout sending {t} to channel, retrying...")
                            await asyncio.sleep(1)
                            try:
                                await context.bot.send_message(chat_id=CHANNEL_ID, text=final_msg, parse_mode='HTML', disable_web_page_preview=True)
                                context.bot_data['channel_mem']['tickers'].append(t)
                                results_found += 1
                            except: pass
                        except telegram.error.RetryAfter as e:
                            logger.warning(f"Rate limited, waiting {e.retry_after} seconds...")
                            await asyncio.sleep(e.retry_after)
                            try:
                                await context.bot.send_message(chat_id=CHANNEL_ID, text=final_msg, parse_mode='HTML', disable_web_page_preview=True)
                                context.bot_data['channel_mem']['tickers'].append(t)
                                results_found += 1
                            except: pass
                        except Exception as e:
                            logger.error(f"Error sending {t} to channel: {e}")
                        
                elif not is_auto:
                    card = format_dashboard_card(t, d, shares, is_new, info, p['risk_usd'], p['sma'], public_view=False)
                    try:
                        await context.bot.send_message(chat_id=target_chat_id, text=card, parse_mode='HTML', disable_web_page_preview=True)
                        results_found += 1
                    except telegram.error.RetryAfter as e:
                        logger.warning(f"Rate limited, waiting {e.retry_after} seconds...")
                        await asyncio.sleep(e.retry_after)
                        try:
                            await context.bot.send_message(chat_id=target_chat_id, text=card, parse_mode='HTML', disable_web_page_preview=True)
                            results_found += 1
                        except: pass
                    except Exception as e:
                        logger.error(f"Error sending {t} to user: {e}")
                
        except Exception as e:
            logger.error(f"Error processing {t}: {e}")
            if manual_mode: 
                try:
                    await context.bot.send_message(target_chat_id, f"⚠️ {t}: {str(e)[:100]}")
                except: pass
            continue
    
    if all_data is not None:
        del all_data
        gc.collect()
    
    if not is_auto:
        context.user_data['scanning'] = False
        
    if status_msg:
        try:
            await status_msg.edit_text(f"✅ <b>{scan_type} Scan Complete.</b>\nFound: {results_found} signals.", parse_mode='HTML')
        except: pass

# ==========================================
# BOT HANDLERS
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
        except Exception as e: 
            print(f"⚠️ Error fetching whitelist: {e}")
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
    
    if 'active_users' not in context.bot_data: 
        context.bot_data['active_users'] = set()
    context.bot_data['active_users'].add(user_id)
    return True

async def safe_get_params(context):
    if 'params' not in context.user_data: 
        context.user_data['params'] = DEFAULT_PARAMS.copy()
    return context.user_data['params']

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
    return ReplyKeyboardMarkup([
        [KeyboardButton("SMA 100"), KeyboardButton("SMA 150"), KeyboardButton("SMA 200")], 
        [KeyboardButton("🔙 Back")]
    ], resize_keyboard=True)

def get_tf_keyboard():
    return ReplyKeyboardMarkup([
        [KeyboardButton("Daily (D)"), KeyboardButton("Weekly (W)")], 
        [KeyboardButton("🔙 Back")]
    ], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update, context): return
    try:
        p = await safe_get_params(context)
    except:
        p = DEFAULT_PARAMS.copy()
        context.user_data['params'] = p

    context.user_data['input_mode'] = None
    
    if context.args and context.args[0] == 'autoscan':
        await update.message.reply_text("🚀 <b>Auto-starting Scan...</b>", parse_mode='HTML')
        context.user_data['scanning'] = True
        tickers = get_sp500_tickers()
        asyncio.create_task(run_scan_process(update, context, p, tickers, manual_mode=False))
        return

    user_name = update.effective_user.first_name
    
    welcome_text = f"""👋 <b>Welcome to the S&P500 Sequence Screener, {user_name}!</b>

I am a quantitative trading bot for <b>S&P 500</b> analysis.

Use the buttons below to configure your scan settings.

<b>Quick Start:</b>
• Press <b>▶️ START SCAN</b> to scan all S&P 500 stocks
• Type a ticker (e.g. <code>AAPL</code>) for diagnostic mode

<i>Press ℹ️ HELP / INFO for full documentation.</i>"""

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
    
    if text == "🔙 Back":
        context.user_data['input_mode'] = None
        await update.message.reply_text("🔙 Main Menu", reply_markup=get_main_keyboard(p))
        return

    if text == "▶️ START SCAN":
        if context.user_data.get('scanning'): 
            return await update.message.reply_text("⚠️ Already running!")
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
        help_text = (
            "<b>📚 S&P500 SCREENER</b>\n\n"
            "<b>Commands:</b>\n"
            "• <b>▶️ START SCAN</b> - Scan all S&P 500\n"
            "• <b>⏹ STOP SCAN</b> - Stop scanning\n"
            "• Type ticker (e.g. <code>AAPL</code>) for diagnostics\n\n"
            "<b>Settings:</b>\n"
            "• 💸 Risk - Position size in $\n"
            "• ⚖️ RR - Minimum Risk/Reward\n"
            "• 📊 ATR - Max volatility filter\n"
            "• 📈 SMA - Trend filter period\n"
        )
        return await update.message.reply_html(help_text)

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
            await update.message.reply_text(f"✅ SMA set to {p['sma']}", reply_markup=get_main_keyboard(p))
        return
        
    if context.user_data.get('input_mode') == "tf_select":
        if "Daily" in text: 
            p['tf'] = "Daily"
        elif "Weekly" in text: 
            p['tf'] = "Weekly"
        context.user_data['input_mode'] = None
        context.user_data['params'] = p
        await update.message.reply_text(f"✅ Timeframe set to {p['tf']}", reply_markup=get_main_keyboard(p))
        return

    if "Only New" in text: 
        p['new_only'] = not p['new_only']
        status = "ENABLED" if p['new_only'] else "DISABLED"
        context.user_data['params'] = p
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
            await update.message.reply_text(f"✅ Risk updated to ${val}", reply_markup=get_main_keyboard(p))
        except: 
            await update.message.reply_text("❌ Invalid amount.")
        return
    elif mode == "rr":
        try:
            val = float(text)
            if val < 0.1: raise ValueError
            p['min_rr'] = val
            context.user_data['input_mode'] = None
            context.user_data['params'] = p
            await update.message.reply_text(f"✅ Min RR updated to {val}", reply_markup=get_main_keyboard(p))
        except: 
            await update.message.reply_text("❌ Invalid number.")
        return
    elif mode == "atr":
        try:
            val = float(text)
            if val < 0.1: raise ValueError
            p['max_atr'] = val
            context.user_data['input_mode'] = None
            context.user_data['params'] = p
            await update.message.reply_text(f"✅ Max ATR updated to {val}%", reply_markup=get_main_keyboard(p))
        except: 
            await update.message.reply_text("❌ Invalid number.")
        return

    # Manual ticker entry
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
# SCHEDULER & CHANNEL SCAN
# ==========================================
async def trigger_channel_scan(app):
    print("🚀 Triggering Channel Scan...")
    
    try:
        await app.bot.send_message(
            chat_id=ADMIN_ID, 
            text="🔄 <b>Auto-Scan Started...</b>",
            parse_mode='HTML'
        )
    except Exception as e: 
        print(f"Notify Error: {e}")

    channel_params = {
        'risk_usd': 100.0, 'min_rr': 1.5, 'max_atr': 5.0, 'sma': 200,           
        'tf': 'Daily', 'new_only': True, 'auto_scan': True
    }
    
    tickers = get_sp500_tickers()
    
    class DummyObj: pass
    u_upd = DummyObj()
    u_upd.effective_chat = None 
    u_ctx = DummyObj()
    u_ctx.bot = app.bot
    u_ctx.user_data = {}
    u_ctx.bot_data = app.bot_data 
    
    await run_scan_process(u_upd, u_ctx, channel_params, tickers, manual_mode=False, is_auto=True)

    try:
        await app.bot.send_message(
            chat_id=ADMIN_ID, 
            text="✅ <b>Auto-Scan Complete.</b>",
            parse_mode='HTML'
        )
    except: pass

async def force_auto_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID: return
    await update.message.reply_text("🚀 <b>Forcing Channel Scan...</b>", parse_mode='HTML')
    await trigger_channel_scan(context.application)

async def auto_scan_scheduler(app):
    print("⏳ Scheduler started... (Target: 15:00 ET)")
    while True:
        try:
            ny_tz = pytz.timezone('US/Eastern')
            now = datetime.datetime.now(ny_tz)
            
            is_market_day = now.weekday() < 5
            is_scan_time = (now.hour == 15 and now.minute == 0)
            
            if is_market_day and is_scan_time:
                print("🚀 Auto-Scan Triggered for CHANNEL!")
                await trigger_channel_scan(app)
                await asyncio.sleep(61)
            
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Scheduler Error: {e}")
            await asyncio.sleep(60)

# ==========================================
# GATEKEEPER (JOIN REQUESTS)
# ==========================================
async def handle_join_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id 
    callback_data = f"agree|{user_id}|{chat_id}"
    
    terms_text = (
        "⚖️ <b>LEGAL DISCLAIMER</b>\n\n"
        "By joining, you agree:\n"
        "1. This is NOT financial advice\n"
        "2. Trading involves risk of loss\n"
        "3. You are responsible for your decisions\n\n"
        "<i>Click below to accept and join:</i>"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ I AGREE", callback_data=callback_data)],
        [InlineKeyboardButton("❌ Decline", callback_data="decline")]
    ])
    
    try:
        await context.bot.send_message(chat_id=user_id, text=terms_text, reply_markup=keyboard, parse_mode='HTML')
    except Exception as e:
        print(f"⚠️ Could not DM user {user_id}: {e}")

async def handle_terms_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "decline":
        await query.edit_message_text("❌ <b>Access Denied.</b>", parse_mode='HTML')
        return

    if data.startswith("agree"):
        try:
            _, user_id, channel_id = data.split("|")
            await context.bot.approve_chat_join_request(chat_id=channel_id, user_id=user_id)
            await query.edit_message_text("✅ <b>Welcome!</b> You've been approved.", parse_mode='HTML')
            await context.bot.send_message(ADMIN_ID, f"👤 New Member: {user_id}")
        except Exception as e:
            await query.edit_message_text(f"⚠️ Error: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("🤖 Starting bot...")
    
    # TEST: Can we reach Telegram at all?
    print("🔍 Testing Telegram connectivity...")
    try:
        test_resp = requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getMe",
            timeout=30
        )
        print(f"✅ Telegram API Response: {test_resp.text[:200]}")
    except Exception as e:
        print(f"❌ Cannot reach Telegram: {e}")
        print("⚠️ Will try anyway...")
    
    # Simple setup with default timeouts
    my_persistence = PicklePersistence(filepath='bot_data.pickle', update_interval=1)
    app = (
        ApplicationBuilder()
        .token(TG_TOKEN)
        .persistence(my_persistence)
        .build()
    )
    
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('stats', stats_command))
    app.add_handler(CommandHandler('auto', force_auto_scan))
    app.add_handler(ChatJoinRequestHandler(handle_join_request))
    app.add_handler(CallbackQueryHandler(handle_terms_callback))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("✅ Bot handlers registered")
    print("🚀 Starting polling...")
    
    # Create scheduler task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_bot():
        async with app:
            await app.start()
            asyncio.create_task(auto_scan_scheduler(app))
            await app.updater.start_polling(drop_pending_updates=True)
            print("✅ Bot is running!")
            
            # Keep running forever
            while True:
                await asyncio.sleep(3600)
    
    loop.run_until_complete(run_bot())

if __name__ == '__main__':
    main()
