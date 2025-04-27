from flask import Flask, render_template
import ccxt
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import logging
import smtplib
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings
settings = {
    'timeframe': '1h',
    'limit': 1000,
    'min_signals': 4,
    'max_pairs': 10,
    'email': os.getenv("EMAIL"),
    'email_password': os.getenv("EMAIL_PASSWORD"),
}

# Validate email settings
if not settings['email'] or not settings['email_password']:
    logger.warning("Email or email password not set. Email alerts will be disabled.")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize exchange
exchange = ccxt.gateio()

# Fetch top trading pairs
def get_top_pairs():
    try:
        markets = exchange.load_markets()
        symbols = [symbol for symbol in markets.keys() if '/USDT' in symbol]
        volumes = []
        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                volumes.append((symbol, ticker['quoteVolume']))
            except ccxt.BaseError as e:
                logger.error(f"Error fetching ticker for {symbol}: {e}")
                continue
        volumes.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in volumes[:settings['max_pairs']]]
    except ccxt.BaseError as e:
        logger.error(f"Error fetching markets: {e}")
        return []

# Fetch OHLCV data
def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=settings['timeframe'], limit=settings['limit'])
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if len(df) < 60: # Minimum data for analysis
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return None
        return df
    except ccxt.BaseError as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

# Fetch news
def fetch_news():
    url = "https://www.reuters.com/markets/cryptocurrencies/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article")[:5]
        news = [{"title": article.find("h3").text.strip()} for article in articles if article.find("h3")]
        return news
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []

# Fetch economic calendar
def fetch_economic_calendar():
    url = "https://www.investing.com/economic-calendar/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        events = []
        table = soup.find("table", {"id": "economicCalendarData"})
        if table:
            for row in table.find("tbody").find_all("tr"):
                cells = row.find_all("td")
                if len(cells) > 5:
                    time = cells[0].text.strip()
                    currency = cells[1].text.strip()
                    impact = cells[2].find("span")["title"] if cells[2].find("span") else "Low"
                    event = cells[3].text.strip()
                    actual = cells[4].text.strip()
                    forecast = cells[5].text.strip()
                    previous = cells[6].text.strip()
                    if "High" in impact or "Medium" in impact:
                        events.append({
                            "time": time,
                            "currency": currency,
                            "impact": impact,
                            "event": event,
                            "actual": actual,
                            "forecast": forecast,
                            "previous": previous
                        })
        return events
    except requests.RequestException as e:
        logger.error(f"Error fetching economic calendar: {e}")
        return []

# Detect continuation candlestick patterns
def detect_continuation_patterns(df):
    if len(df) < 5:
        return []
    patterns = []
    for i in range(len(df) - 5):
        candle = df.iloc[i]
        body = abs(candle['close'] - candle['open'])
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        if body > 0 and upper_shadow < 0.05 * body and lower_shadow < 0.05 * body:
            if candle['close'] > candle['open']:
                patterns.append({"index": i, "pattern": "Bullish Marubozu", "type": "Continuation"})
            else:
                patterns.append({"index": i, "pattern": "Bearish Marubozu", "type": "Continuation"})
        if i <= len(df) - 5:
            c1, c2, c3, c4, c5 = df.iloc[i:i+5]
            if (c1['close'] > c1['open'] and
                c2['close'] < c2['open'] and c3['close'] < c3['open'] and c4['close'] < c4['open'] and
                c5['close'] > c5['open'] and c5['close'] > c1['high']):
                patterns.append({"index": i+4, "pattern": "Rising Three Methods", "type": "Continuation"})
            if (c1['close'] < c1['open'] and
                c2['close'] > c2['open'] and c3['close'] > c3['open'] and c4['close'] > c4['open'] and
                c5['close'] < c5['open'] and c5['close'] < c1['low']):
                patterns.append({"index": i+4, "pattern": "Falling Three Methods", "type": "Continuation"})
        if i <= len(df) - 2:
            c1, c2 = df.iloc[i:i+2]
            if (c1['close'] < c1['open'] and c2['close'] > c2['open'] and
                abs(c2['open'] - c1['close']) < 0.01 * c1['close']):
                patterns.append({"index": i+1, "pattern": "Bullish Separating Lines", "type": "Continuation"})
            if (c1['close'] > c1['open'] and c2['close'] < c2['open'] and
                abs(c2['open'] - c1['close']) < 0.01 * c1['close']):
                patterns.append({"index": i+1, "pattern": "Bearish Separating Lines", "type": "Continuation"})
    return patterns

# Detect Wyckoff phases
def detect_wyckoff_phases(df):
    if len(df) < 50:
        return []
    phases = []
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    
    for i in range(50, len(df) - 1):
        if (df['bb_width'].iloc[i] < 0.1 and
            df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
            df['close'].iloc[i] > df['close'].iloc[i-50:i].min()):
            phases.append({"index": i, "phase": "Accumulation", "signal": "Buy"})
        elif (df['close'].iloc[i] > df['close'].iloc[i-20:i].max() and
              df['volume'].iloc[i] > df['volume_ma'].iloc[i]):
            phases.append({"index": i, "phase": "Markup", "signal": "Buy"})
        elif (df['bb_width'].iloc[i] < 0.1 and
              df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
              df['close'].iloc[i] < df['close'].iloc[i-50:i].max()):
            phases.append({"index": i, "phase": "Distribution", "signal": "Sell"})
        elif (df['close'].iloc[i] < df['close'].iloc[i-20:i].min() and
              df['volume'].iloc[i] > df['volume_ma'].iloc[i]):
            phases.append({"index": i, "phase": "Markdown", "signal": "Sell"})
    
    return phases

# Detect market structure
def detect_market_structure(df):
    if len(df) < 10:
        return []
    structure = []
    df['swing_high'] = df['high'].rolling(window=5, center=True).max()
    df['swing_low'] = df['low'].rolling(window=5, center=True).min()
    
    trend = "Unknown"
    last_high = last_low = None
    for i in range(10, len(df) - 1):
        current_high = df['swing_high'].iloc[i]
        current_low = df['swing_low'].iloc[i]
        if pd.notna(current_high) and pd.notna(current_low):
            if last_high and last_low:
                if current_high > last_high and current_low > last_low:
                    trend = "Bullish"
                    structure.append({"index": i, "structure": "Bullish", "signal": "Buy"})
                elif current_high < last_high and current_low < last_low:
                    trend = "Bearish"
                    structure.append({"index": i, "structure": "Bearish", "signal": "Sell"})
                elif trend == "Bullish" and current_low < last_low:
                    structure.append({"index": i, "structure": "CHoCH Bearish", "signal": "Sell"})
                    trend = "Bearish"
                elif trend == "Bearish" and current_high > last_high:
                    structure.append({"index": i, "structure": "CHoCH Bullish", "signal": "Buy"})
                    trend = "Bullish"
            last_high = current_high
            last_low = current_low
    
    return structure

# Identify trading opportunities
def identify_opportunities(pairs):
    opportunities = []
    
    # Pre-train models (optional, can be enhanced further)
    rf_model = None
    lstm_model = None
    
    def process_symbol(symbol):
        try:
            df = fetch_ohlcv(symbol)
            if df is None:
                return None
            
            # Technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            df['ma50'] = df['close'].rolling(window=50).mean()
            df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['cvd'] = (df['volume'] * (df['close'] - df['open'])).cumsum()
            df['volume_change'] = df['volume'].pct_change() * 100
            
            # LSTM Prediction
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[['close']].values)
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)
            if len(X) > 0:
                nonlocal lstm_model
                if lstm_model is None:
                    lstm_model = Sequential()
                    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
                    lstm_model.add(LSTM(50))
                    lstm_model.add(Dense(1))
                    lstm_model.compile(optimizer='adam', loss='mse')
                    lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                lstm_pred = lstm_model.predict(X[-1].reshape(1, X.shape[1], 1), verbose=0)
                lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
            else:
                lstm_pred = df['close'].iloc[-1]
            
            # Random Forest Prediction
            features = df[['rsi', 'macd', 'stoch', 'adx', 'cvd']].dropna()
            if len(features) > 10:
                nonlocal rf_model
                X = features[:-1]
                y = (df['close'].shift(-1) > df['close']).astype(int)[:-1]
                if rf_model is None:
                    rf_model = RandomForestClassifier(n_estimators=100)
                    rf_model.fit(X, y)
                rf_pred = rf_model.predict(features[-1:])
                rf_signal = "Buy" if rf_pred[0] == 1 else "Sell"
            else:
                rf_signal = "Hold"
            
            # Detect patterns and structures
            continuation_patterns = detect_continuation_patterns(df)
            latest_patterns = [p for p in continuation_patterns if p['index'] >= len(df) - 5]
            wyckoff_phases = detect_wyckoff_phases(df)
            latest_wyckoff = [p for p in wyckoff_phases if p['index'] >= len(df) - 5]
            market_structure = detect_market_structure(df)
            latest_structure = [s for s in market_structure if s['index'] >= len(df) - 5]
            
            # Economic calendar
            economic_events = fetch_economic_calendar()
            event_warning = None
            current_time = datetime.now(timezone.utc)
            for event in economic_events:
                event_time_str = event["time"]
                try:
                    event_time = datetime.strptime(f"{current_time.date()} {event_time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                    time_diff = (event_time - current_time).total_seconds() / 3600
                    if 0 <= time_diff <= 1:
                        event_warning = f"Warning: Upcoming economic event ({event['event']}) at {event_time_str} with {event['impact']} impact"
                        break
                except ValueError:
                    continue
            
            # Identify opportunity
            latest = df.iloc[-1]
            opp = {
                'symbol': symbol,
                'price': latest['close'],
                'timestamp': latest['timestamp'],
                'df': df
            }
            
            # Calculate signals
            buy_signals = 0
            sell_signals = 0
            
            if latest['rsi'] < 30:
                buy_signals += 1
            elif latest['rsi'] > 70:
                sell_signals += 1
                
            if latest['macd'] > latest['macd_signal'] and latest['macd'] < 0:
                buy_signals += 1
            elif latest['macd'] < latest['macd_signal'] and latest['macd'] > 0:
                sell_signals += 1
                
            if latest['close'] < latest['bb_lower']:
                buy_signals += 1
            elif latest['close'] > latest['bb_upper']:
                sell_signals += 1
                
            if latest['stoch'] < 20:
                buy_signals += 1
            elif latest['stoch'] > 80:
                sell_signals += 1
                
            if latest['close'] > latest['ichimoku_a'] and latest['close'] > latest['ichimoku_b']:
                buy_signals += 1
            elif latest['close'] < latest['ichimoku_a'] and latest['close'] < latest['ichimoku_b']:
                sell_signals += 1
                
            if latest['adx'] > 25 and latest['close'] > latest['ma50']:
                buy_signals += 1
            elif latest['adx'] > 25 and latest['close'] < latest['ma50']:
                sell_signals += 1
                
            if latest['ema20'] > latest['ma50']:
                buy_signals += 1
            elif latest['ema20'] < latest['ma50']:
                sell_signals += 1
                
            if latest['cvd'] > df['cvd'].mean():
                buy_signals += 1
            else:
                sell_signals += 1
                
            if latest['volume_change'] > 50:
                if latest['close'] > latest['open']:
                    buy_signals += 1
                else:
                    sell_signals += 1
            
            if lstm_pred > latest['close']:
                buy_signals += 1
            else:
                sell_signals += 1
                
            if rf_signal == "Buy":
                buy_signals += 1
            elif rf_signal == "Sell":
                sell_signals += 1
            
            for pattern in latest_patterns:
                if pattern['pattern'] in ["Bullish Marubozu", "Rising Three Methods", "Bullish Separating Lines"]:
                    buy_signals += 1
                    opp['continuation_pattern'] = pattern['pattern']
                elif pattern['pattern'] in ["Bearish Marubozu", "Falling Three Methods", "Bearish Separating Lines"]:
                    sell_signals += 1
                    opp['continuation_pattern'] = pattern['pattern']
            
            for phase in latest_wyckoff:
                if phase['signal'] == "Buy":
                    buy_signals += 2
                    opp['wyckoff_phase'] = phase['phase']
                elif phase['signal'] == "Sell":
                    sell_signals += 2
                    opp['wyckoff_phase'] = phase['phase']
            
            for struct in latest_structure:
                if struct['signal'] == "Buy":
                    buy_signals += 2
                    opp['market_structure'] = struct['structure']
                elif struct['signal'] == "Sell":
                    sell_signals += 2
                    opp['market_structure'] = struct['structure']
            
            # Calculate success rate
            total_signals = buy_signals + sell_signals
            buy_success = (buy_signals / total_signals * 100) if total_signals > 0 else 0
            sell_success = (sell_signals / total_signals * 100) if total_signals > 0 else 0
            
            # Add strong opportunities
            if buy_success > 85 or sell_success > 85:
                opp['buy_success'] = round(buy_success, 2)
                opp['sell_success'] = round(sell_success, 2)
                if event_warning:
                    opp['event_warning'] = event_warning
                if buy_success > 85:
                    send_email_alert(
                        f"Strong Buy Opportunity: {symbol}",
                        f"Price: {latest['close']}, Success Rate: {buy_success}%, Time: {latest['timestamp']}"
                        f"{f', {event_warning}' if event_warning else ''}"
                        f"{f', Wyckoff Phase: {opp['wyckoff_phase']}' if 'wyckoff_phase' in opp else ''}"
                        f"{f', Market Structure: {opp['market_structure']}' if 'market_structure' in opp else ''}"
                    )
                if sell_success > 85:
                    send_email_alert(
                        f"Strong Sell Opportunity: {symbol}",
                        f"Price: {latest['close']}, Success Rate: {sell_success}%, Time: {latest['timestamp']}"
                        f"{f', {event_warning}' if event_warning else ''}"
                        f"{f', Wyckoff Phase: {opp['wyckoff_phase']}' if 'wyckoff_phase' in opp else ''}"
                        f"{f', Market Structure: {opp['market_structure']}' if 'market_structure' in opp else ''}"
                    )
                return opp
            return None
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    # Process symbols in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_symbol, pairs)
        opportunities = [opp for opp in results if opp is not None]
    
    return opportunities

# Send email alerts
def send_email_alert(subject, body):
    if not settings['email'] or not settings['email_password']:
        logger.warning("Cannot send email: Email settings not configured")
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = settings['email']
    msg['To'] = settings['email']
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(settings['email'], settings['email_password'])
            server.sendmail(settings['email'], settings['email'], msg.as_string())
        logger.info(f"Email sent: {subject}")
    except smtplib.SMTPException as e:
        logger.error(f"Error sending email: {e}")

# Create chart
def create_chart(opp):
    df = opp['df']
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma50'], name='MA50', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema20'], name='EMA20', line=dict(color='orange')))
    fig.update_layout(title=f"{opp['symbol']} Chart", xaxis_title="Time", yaxis_title="Price")
    return json.dumps(fig, cls=PlotlyJSONEncoder)

# Main route
@app.route('/')
def index():
    try:
        pairs = get_top_pairs()
        if not pairs:
            return render_template('error.html', message="No trading pairs available"), 500
        opportunities = identify_opportunities(pairs)
        news = fetch_news()
        economic_events = fetch_economic_calendar()
        charts = {opp['symbol']: create_chart(opp) for opp in opportunities}
        return render_template('index.html', opportunities=opportunities, charts=charts, news=news, economic_events=economic_events)
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return render_template('error.html', message="An error occurred while processing the request"), 500

# Run the app
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)