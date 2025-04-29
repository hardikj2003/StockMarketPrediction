import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from prophet import Prophet
from prophet.plot import plot_plotly

##########################################################################################
## PART 1: Define Functions for Pulling, Processing, Creating Technical Indicators, and Forecasting ##
##########################################################################################

# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

# Process data to ensure it is timezone-aware and has the correct format
def process_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    # if data.index.tzinfo is None:
    #     data.index = data.index.tz_localize('UTC')
    # data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True)
    return data

# Calculate basic metrics from the stock data
def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

# Forecast stock prices using Prophet
def forecast_stock_prices(data, forecast_period_days):
    # Create a dataframe for Prophet
    df_prophet = data[['Datetime', 'Close']].rename(columns={
        'Datetime': 'ds',
        'Close': 'y'
    })
    
    # Fit Prophet model
    m = Prophet()
    m.fit(df_prophet)
    
    # Create future dataframe
    future = m.make_future_dataframe(periods=forecast_period_days)
    
    # Make predictions
    forecast = m.predict(future)
    
    return forecast, m

###############################################
## PART 2: Creating the Dashboard App layout ##
###############################################

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Real Time Stock Dashboard with Forecasting')

# 2A: SIDEBAR PARAMETERS ############
# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'ADBE')
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

# Forecasting parameters
st.sidebar.header('Forecasting Parameters')
enable_forecasting = st.sidebar.checkbox('Enable Forecasting', value=True)
forecast_years = st.sidebar.slider('Forecast Period (Years)', 1, 4, 1) if enable_forecasting else 0
forecast_days = forecast_years * 365 if enable_forecasting else 0

# Mapping of time periods to data intervals
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# 2B: MAIN CONTENT AREA ############
# Update the dashboard based on user input
if st.sidebar.button('Update'):
    # Fetch and process data
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    data = process_data(data)
    
    # Calculate metrics
    last_close, change, pct_change, high, low, volume = calculate_metrics(data)
    
    # Ensure scalar values
    last_close = float(last_close.item()) if hasattr(last_close, 'item') else float(last_close)
    change = float(change.item()) if hasattr(change, 'item') else float(change)
    pct_change = float(pct_change.item()) if hasattr(pct_change, 'item') else float(pct_change)
    high = float(high.item()) if hasattr(high, 'item') else float(high)
    low = float(low.item()) if hasattr(low, 'item') else float(low)
    volume = float(volume.item()) if hasattr(volume, 'item') else float(volume)
    
    # Display main metrics
    st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")
    
    # Plot the stock price chart
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'], 
            open=data['Open'], 
            high=data['High'], 
            low=data['Low'], 
            close=data['Close'],
            name='Historical Data'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data['Datetime'],
            y=data['Close'],
            mode='lines',
            name='Historical Data'
        ))
    
    # Format graph
    fig.update_layout(
        title=f'{ticker} {time_period.upper()} Chart',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=600
    )
    
    # Forecast and display if enabled
    if enable_forecasting:
        # For forecasting, get daily data regardless of user selected interval
        forecast_data = fetch_stock_data(ticker, '5y', '1d')
        forecast_data = process_data(forecast_data)
        
        if not forecast_data.empty:
            try:
                st.subheader(f'Price Forecast for {ticker} ({forecast_years} years)')
                
                # Run forecasting
                forecast, model = forecast_stock_prices(forecast_data, forecast_days)
                
                # Show forecast components
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(
                    title=f'{ticker} Forecast for {forecast_years} Years',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Show forecast components
                st.subheader('Forecast Components')
                fig_components = model.plot_components(forecast)
                st.write(fig_components)
                
                # Display forecast data
                st.subheader('Forecast Data (Last 10 days)')
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
                
            except Exception as e:
                st.error(f"Error in forecasting: {str(e)}")
                st.info("Forecasting requires clean daily data. Try a different ticker or disable forecasting.")
    
    # Display the historical chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display historical data and technical indicators
    st.subheader('Historical Data')
    st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

# 2C: SIDEBAR PRICES ############
# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']

for symbol in stock_symbols:
    try:
        real_time_data = fetch_stock_data(symbol, '1d', '1m')
        if not real_time_data.empty:
            real_time_data = process_data(real_time_data)
            
            # Ensure 'Close' and 'Open' columns exist and are not empty
            if 'Close' in real_time_data.columns and 'Open' in real_time_data.columns:
                if not real_time_data['Close'].empty and not real_time_data['Open'].empty:
                    # Convert to scalar values
                    last_price = real_time_data['Close'].iloc[-1]
                    open_price = real_time_data['Open'].iloc[0]
                    
                    # Ensure they're scalar
                    last_price = float(last_price.item()) if hasattr(last_price, 'item') else float(last_price)
                    open_price = float(open_price.item()) if hasattr(open_price, 'item') else float(open_price)
                    
                    change = last_price - open_price
                    pct_change = (change / open_price) * 100
                    
                    st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")
    except Exception as e:
        st.sidebar.warning(f"Could not fetch data for {symbol}")

# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides stock data, technical indicators, and price forecasting for various time periods. Use the sidebar to customize your view.')