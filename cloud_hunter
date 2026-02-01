import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import google.generativeai as genai

# ==============================================================================
# AYARLAR
# ==============================================================================
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_API_TOKEN = os.environ.get("PUSHOVER_API_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

WATCH_TIMEFRAMES = ["1h", "4h"] 

# GÖSTERGE AYARLARI
DEMA_LEN = 200
ATR_LEN = 14
ST_FACTOR = 3.0
ADX_LEN = 14
LAZY_MA = 34
LAZY_SIG = 9
FRESHNESS = 3 

# GALAXY LIST (SADECE ABD)
TARGET_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "AVGO", "NFLX",
    "INTC", "QCOM", "CSCO", "DELL", "APP", "TSM", "BIDU", "BABA", "PLTR", "CRWD",
    "RBRK", "LSCC", "BBAI", "ZM", "ZS", "ZETA", "CLS", "PENG", "SOXL",
    "ADBE", "CRM", "NOW", "ORCL", "IBM", "INTU", "UBER", "ABNB", "BKNG", "PANW",
    "FTNT", "SNOW", "SQ", "SHOP", "U", "ROKU", "DKNG", "HOOD", "PYPL", "MU", "TXN",
    "LRCX", "ADI", "KLAC", "ARM", "SMCI", "SNDK", "AMAT", "ON", "MCHP", "CDNS", "SNPS",
    "DDOG", "NET", "MDB", "TEAM", "TTWO", "EA", "PDD", "JD", "OKTA",
    "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "SCHW", "USB",
    "TRV", "AIG", "SPGI", "COIN", "MSTR", "BRK-B", "PGR", "CB", "CME", "ICE", "COF", "SYF",
    "BA", "GE", "F", "GM", "CAT", "DE", "HON", "UNP", "UPS", "FDX", "LMT", "RTX",
    "NOC", "GD", "EMR", "MMM", "ETN", "VZ", "T", "TMUS", "CMCSA", "ADP", "CSX", "NSC",
    "WM", "RSG", "RIVN", "LCID",
    "JNJ", "PFE", "MRNA", "REGN", "LLY", "UNH", "ABBV", "AMGN", "BMY", "GILD", "ISRG",
    "SYK", "CVS", "TMO", "DHR", "VRTX", "MOH", "MDT", "BSX", "ZTS", "CI", "HUM",
    "WMT", "COST", "PG", "KO", "PEP", "XOM", "CVX", "DIS", "MCD", "NKE", "SBUX",
    "TGT", "LOW", "HD", "TJX", "LULU", "MDLZ", "PM", "MO", "CL", "KMB", "EL",
    "CMG", "MAR", "KHC", "HSY", "KR",
    "OXY", "SLB", "HAL", "COP", "EOG", "FCX", "NEM", "LIN", "DOW", "SHW", "NEE",
    "DUK", "SO", "MPC", "APD", "ECL", "NUE", "PLD", "AMT", "CCI", "EQIX", "PSA",
    "NVDX", "AAPU", "GGLL", "AMZZ", "METU", "AMZP", "MARA", "QQQT",
    "O", "AGNC", "ORC", "SPHD", "DX", "OXLC", "GLAD", "GAIN", "GOOD", "LAND", "SRET",
    "QYLD", "XYLD", "SDIV", "DIV", "RYLD", "JEPI", "JEPQ", "EFC", "SCM", "PSEC",
    "QQQY", "APLE", "MAIN", "WSR", "ARR", "SBR", "GROW", "HRZN", "LTC", "PNNT",
    "SLG", "ARCC", "HTGC", "SPG", "NLY", "ETV", "PDI", "ARE", "FRT", "SPYI", "WPC",
    "ECC", "OMAH", "QQQI", "ABR", "IIPR", "CIM", "VNM", "RIET", "DLR", "VICI", "OXSQ",
    "OMCL", "POWL", "DXPE", "TLN", "RH", "TOST", "NU", "MOS", "AES", "ASRT", "WRD",
    "CRS", "LUV", "ALL", "AYI", "APTV", "BIIB", "FTI", "VERU", "AZO", "CEG", "NVO",
    "MRK",
    "AGQ", "UGL", "LIT", "QQQ", "TQQQ", "UAMY", "WEAT", "GOOP", "QLD", "YINN",
    "IGM", "SPY", "PFIX", "TLT", "TLTW", "BIL", "VOO", "VTI", "BND", "VYM", "SCHD"
]

# ==============================================================================
# HESAPLAMA MOTORU & RESAMPLING
# ==============================================================================

def resample_custom_us_4h(df_1h):
    if df_1h.empty: return df_1h
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df_1h.index = df_1h.index.tz_convert('America/New_York')

    df_1h = df_1h.between_time('09:30', '16:00')
    agg_candles = []
    
    for date, group in df_1h.groupby(df_1h.index.date):
        session1 = group[group.index.hour < 13] 
        if not session1.empty:
            agg_candles.append({
                'time': session1.index[0],
                'open': session1['open'].iloc[0],
                'high': session1['high'].max(),
                'low': session1['low'].min(),
                'close': session1['close'].iloc[-1], 
                'volume': session1['volume'].sum()
            })
        session2 = group[group.index.hour >= 13]
        if not session2.empty:
            agg_candles.append({
                'time': session2.index[0],
                'open': session2['open'].iloc[0],
                'high': session2['high'].max(),
                'low': session2['low'].min(),
                'close': session2['close'].iloc[-1], 
                'volume': session2['volume'].sum()
            })
            
    if not agg_candles: return pd.DataFrame()
    df_4h = pd.DataFrame(agg_candles)
    df_4h.set_index('time', inplace=True)
    return df_4h

def calculate_dema(series, length):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2 * ema1 - ema2

def calculate_smma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_lazybear_macd(df, len_ma, len_sig):
    src = (df['high'] + df['low'] + df['close']) / 3
    impulse_macd = calculate_smma(src, len_ma)
    impulse_signal = calculate_smma(impulse_macd, len_sig)
    df['LB_Macd'] = impulse_macd
    df['LB_Signal'] = impulse_signal
    df['LB_Cross'] = (df['LB_Macd'] > df['LB_Signal']) & (df['LB_Macd'].shift(1) < df['LB_Signal'].shift(1))
    return df

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up = hl2 - (multiplier * atr)
    dn = hl2 + (multiplier * atr)
    trend = np.ones(len(df))
    trend_up = np.zeros(len(df))
    trend_dn = np.zeros(len(df))
    close = df['close'].values
    up_val = up.values
    dn_val = dn.values
    trend_up[0] = up_val[0]
    trend_dn[0] = dn_val[0]
    for i in range(1, len(df)):
        if close[i-1] > trend_up[i-1]:
            trend_up[i] = max(up_val[i], trend_up[i-1])
        else:
            trend_up[i] = up_val[i]
        if close[i-1] < trend_dn[i-1]:
            trend_dn[i] = min(dn_val[i], trend_dn[i-1])
        else:
            trend_dn[i] = dn_val[i]
        if close[i] > trend_dn[i-1]:
            trend[i] = 1
        elif close[i] < trend_up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    df['ST_Trend'] = trend
    return df

def calculate_adx(df, length=14):
    df = df.copy()
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    df['tr_smooth'] = df['tr'].ewm(alpha=1/length, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/length, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/length, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=1/length, adjust=False).mean()
    return df

def analyze_stock(symbol, tf):
    try:
        fetch_interval = "1h" 
        fetch_period = "1y" 
        
        df = yf.download(symbol, period=fetch_period, interval=fetch_interval, progress=False, auto_adjust=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns=lambda x: x.lower(), inplace=True)

        if tf == "4h":
            df = resample_custom_us_4h(df)
        
        if df.empty or len(df) < (DEMA_LEN + 20): return None

        df['DEMA'] = calculate_dema(df['close'], DEMA_LEN)
        df = calculate_supertrend(df, ATR_LEN, ST_FACTOR)
        df = calculate_adx(df, ADX_LEN)
        df = calculate_lazybear_macd(df, LAZY_MA, LAZY_SIG)

        current = df.iloc[-1]
        
        # --- FİLTRELER ---
        if current['ST_Trend'] != 1: return None 
        
        # TAZELİK KONTROLÜ
        lookback = int(FRESHNESS) + 1
        recent_trends = df['ST_Trend'].tail(lookback).values
        
        trend_changed_recently = False
        for i in range(len(recent_trends)-1, 0, -1):
            if recent_trends[i] == 1 and recent_trends[i-1] == -1:
                trend_changed_recently = True
                break
        
        if not trend_changed_recently: return None

        # LazyBear Durumu
        recent_crosses = df['LB_Cross'].tail(lookback).values
        lb_status = "LB Nötr"
        if True in recent_crosses:
             lb_status = "⚡ LB Cross"

        return {
            "symbol": symbol,
            "tf": tf.upper(),
            "price": round(current['close'], 2),
            "dema": round(current['DEMA'], 2),
            "adx": round(current['adx'], 2),
            "lb_status": lb_status
        }

    except Exception as e:
        return None

# ==============================================================================
# İLETİŞİM MOTORU
# ==============================================================================

def get_gemini_summary(signals):
    if not GEMINI_API_KEY: return "Gemini API Key eksik."
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    Sen 'TrendHunter' adında borsa asistanısın. Kullanıcıya 'Başkan' de.
    Aşağıda yakalanan YENİ sinyaller var.
    
    Görevlerin:
    1. Hisseleri gruplayarak listele.
    2. Her hisse için: Fiyatın DEMA (200) seviyesine göre konumunu yorumla. 
       (Fiyat > DEMA ise 'Güvenli Bölge', Fiyat < DEMA ise 'Riskli/Tepki' gibi).
    3. LazyBear durumu 'LB Cross' ise momentumun da desteklediğini belirt.
    4. Türkçe konuş, emojiler kullan.
    
    Sinyaller:
    {signals}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Hatası: {e}"

def send_pushover(message):
    if not PUSHOVER_USER_KEY or not PUSHOVER_API_TOKEN:
        print("Pushover ayarları eksik.")
        return

    url = "https://api.pushover.net/1/messages.json"
    payload = {
        "token": PUSHOVER_API_TOKEN,
        "user": PUSHOVER_USER_KEY,
        "message": message,
        "title": "🚀 TREND HUNTER ALARM",
        "sound": "cashregister", 
        "priority": 1 
    }
    
    try:
        requests.post(url, data=payload)
        print("✅ Pushover Bildirimi Gönderildi!")
    except Exception as e:
        print(f"Pushover Gönderim Hatası: {e}")

if __name__ == "__main__":
    print("🚀 CLOUD HUNTER V5 BAŞLATILIYOR...")
    all_signals = []
    
    for tf in WATCH_TIMEFRAMES:
        print(f"⏱️ {tf.upper()} Taraması...")
        for sym in TARGET_SYMBOLS:
            time.sleep(0.2)
            res = analyze_stock(sym, tf)
            if res:
                print(f"   🔥 Sinyal ({tf}): {sym}")
                all_signals.append(res)
    
    if all_signals:
        print(f"🎉 Sinyal bulundu. Raporlanıyor...")
        ai_msg = get_gemini_summary(all_signals)
        send_pushover(ai_msg)
    else:
        print("Sinyal yok.")
