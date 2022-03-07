import ta
import ast
import time
import math
import talib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from tensorflow.keras.layers import Dense, Dropout, LSTM
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

#Configs
st.set_page_config(layout='wide', initial_sidebar_state="expanded")
yf.pdr_override()

# Main Code Area Begins
st.title("Stock Market Screener And Price Predictor")
st.subheader('A Platform to test your Trading Strategies, Create Your Portfolio and Learn about Stock market')
def stockList():
    df = pd.read_csv('dataset/nse_stocks.csv')
    dicts = {}
    for i in range(len(df)):
        dicts[df['Company Name'][i]] = df['Symbols'][i]
    return dicts

dicts = stockList()
stock_list = dicts.keys()

st.sidebar.title('Select Stocks From NSE/BSE Dataset')
index = option = st.sidebar.selectbox('Select Stock Market Index', ('NSE', 'BSE', 'Stocks in NSE Sectors'))
if index == 'NSE':
    def stockList():
        df = pd.read_csv('dataset/nse_stocks.csv')
        dicts = {}
        for i in range(len(df)):
            dicts[df['Company Name'][i]] = df['Symbols'][i]
        return dicts
    dicts = stockList()
    stock_list = dicts.keys()
if index == 'BSE':
    def stockList():
        df = pd.read_csv('dataset/bse_stocks.csv')
        dicts = {}
        for i in range(len(df)):
            dicts[df['Company Name'][i]] = df['Symbols'][i]
        return dicts
    dicts = stockList()
    stock_list = dicts.keys()
if index == 'Stocks in NSE Sectors':
    sector = st.sidebar.selectbox('Select Stocks from Nifty Sectors',
                                  ('Auto Manufacturing', 'Energy', 'FMCG', 'Finance Service', 'IT', 'Media', 'Metal', 'PSU Banks', 'Private Banks', 'Pharmaceuticals', 'Realty'))

    def stockList(sector):
        if sector == 'Auto Manufacturing':
            df = pd.read_csv('dataset/niftyauto.csv')
        elif sector == 'Energy':
            df = pd.read_csv('dataset/niftyenergy.csv')
        elif sector == 'FMCG':
            df = pd.read_csv('dataset/niftyfmcg.csv')
        elif sector == 'Finance Service':
            df = pd.read_csv('dataset/niftyfinserv.csv')
        elif sector == 'IT':
            df = pd.read_csv('dataset/niftyit.csv')
        elif sector == 'Media':
            df = pd.read_csv('dataset/niftymedia.csv')
        elif sector == 'Metal':
            df = pd.read_csv('dataset/niftymetal.csv')
        elif sector == 'PSU Banks':
            df = pd.read_csv('dataset/niftypsubank.csv')
        elif sector == 'Private Banks':
            df = pd.read_csv('dataset/niftyprvtbanks')
        elif sector == 'Pharmaceuticals':
            df = pd.read_csv('dataset/niftypharma.csv')
        elif sector == 'Realty':
            df = pd.read_csv('dataset/niftyrealty.csv')
        else:
            print('Error in Code!')
        dicts = {}
        for i in range(len(df)):
            dicts[df['Company Name'][i]] = df['Symbols'][i]
        return dicts, sector
    dicts, sector = stockList(sector=sector)
    stock_list = dicts.keys()
if index == 'NSE' or index == 'BSE':
    stock = st.sidebar.selectbox(f'Select Stock from {index}', stock_list)
    StockName = stock
    stock = dicts.get(stock)
else:
    stock = st.sidebar.selectbox(
        f'Select Stock from NSE {sector} Sector', stock_list)
    StockName = stock
    stock = dicts.get(stock)

period_list = {'6 Month': '6mo', '1 Year': '1Y', '2 Years': '2Y', '5 Years': '5Y', '10 Years': '10Y', 'Max': 'max'}
periods = period_list.keys()
period = st.sidebar.selectbox('Select the Time period of stocks Data?', periods)
period = period_list.get(period)

df = yf.download(stock, period=period)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(df)
dwn_label = f'Download Dataset for {StockName} to your Machine'
st.sidebar.download_button(label=dwn_label, data=csv, file_name=f'{StockName}.csv', mime='text/csv')

showindex = st.sidebar.checkbox('Show Nifty and Bank Nifty Charts')
if showindex:
    with st.expander('NIFTY Chart'):
        nifty = yf.download('^NSEI', period=period)

        def nifty_pred():
            model = load_model('model/nifty.h5')
            #Creating a new dataframe
            new_df = nifty.filter(['Close'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(new_df)

            #Getting the last 60 days closing price Values and converting the data frame to an array
            last_60_days = new_df[-60:].values

            #Scale the data to be the values between 0 and 1
            last_60_days_scaled = scaler.transform(last_60_days)

            #Create an empty List
            X_test = []

            #Append the past 60 days
            X_test.append(last_60_days_scaled)

            #Convert the X_test data set to a numpy array
            X_test = np.array(X_test)

            #Reshape the data
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            #Get the predicted scaled price
            pred_price = model.predict(X_test)

            #Undoing the Scaling
            pred_price = scaler.inverse_transform(pred_price)
            st.write(f"Price for Tomorrow's Nifty Index :\n {pred_price}")

        fig = go.Figure(data=[go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name='Price')])
        layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Nifty', height=800)
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
        nifty_price = st.button("Get Price Quote for Tommorrow's Nifty")
        if nifty_price:
            nifty_pred()

    with st.expander('BANK NIFTY Chart'):
        bank = yf.download('^NSEBANK', period=period)

        def bank_pred():
            model = load_model('model/nifty.h5')
            #Creating a new dataframe
            new_df = bank.filter(['Close'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(new_df)

            #Getting the last 60 days closing price Values and converting the data frame to an array
            last_60_days = new_df[-60:].values

            #Scale the data to be the values between 0 and 1
            last_60_days_scaled = scaler.transform(last_60_days)

            #Create an empty List
            X_test = []

            #Append the past 60 days
            X_test.append(last_60_days_scaled)

            #Convert the X_test data set to a numpy array
            X_test = np.array(X_test)

            #Reshape the data
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            #Get the predicted scaled price
            pred_price = model.predict(X_test)

            #Undoing the Scaling
            pred_price = scaler.inverse_transform(pred_price)
            st.write(f"Price for Tomorrow's Bank Nifty Index :\n {pred_price}")

        fig = go.Figure(data=[go.Candlestick(x=bank.index, open=bank['Open'], high=bank['High'], low=bank['Low'], close=bank['Close'], name='Price')])
        layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Bank Nifty', height=800)
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
        price = st.checkbox("Get Price Quote for Tommorrow's Bank Nifty")
        if price:
            bank_pred()

#Screener Section
def screen_stocks(nsescreen, period):
    if nsescreen == 'NIFTY 50':
        data = pd.read_csv('dataset/ind_nifty50list.csv')
        index_name = '^NSEI'
    elif nsescreen == 'NIFTY 100':
        data = pd.read_csv('dataset/ind_nifty100list.csv')
        index_name = '^CNX100'
    elif nsescreen == 'NIFTY 200':
        data = pd.read_csv('dataset/ind_nifty200list.csv')
        index_name = '^CNX200'
    elif nsescreen == 'NIFTY 500':
        data = pd.read_csv('dataset/ind_nifty200list.csv')
        index_name = '^CRSLDX'
    elif nsescreen == 'NIFTY Midcap 50':
        data = pd.read_csv('dataset/ind_niftymidcap50list.csv')
        index_name = '^NSEMDCP50'
    elif nsescreen == 'NIFTY Midcap 100':
        data = pd.read_csv('dataset/ind_niftymidcap100list.csv')
        index_name = 'NIFTY_MIDCAP_100.NS'
    elif nsescreen == 'NIFTY Smallcap 50':
        data = pd.read_csv('dataset/ind_niftysmallcap50list.csv')
        index_name = '^NSEI'
    elif nsescreen == 'NIFTY Smallcap 250':
        data = pd.read_csv('dataset/ind_niftysmallcap250list.csv')
        index_name = '^NSEI'
    else:
        st.write('Error in Code!')

    final_data = pd.DataFrame(data, columns=['Company Name', 'Symbol'])

    exportList = pd.DataFrame(columns=['Stock', "Relative Strength Rating", "Current Price", "50 Day EMA", "150 Day EMA", "200 Day EMA", "52 Week Low", "52 week High"])
    returns_multiples = []
    nifty_50 = []
    for i in range(len(final_data)):
        nifty_50.append(final_data['Symbol'][i]+'.NS')

    #Nifty 50 index Returns
    index_df = yf.download(index_name, period=period)
    index_df['Percent Change'] = index_df['Adj Close'].pct_change()
    index_return = (index_df['Percent Change'] + 1).cumprod()[-1]

    with st.spinner(f'Screening Stock from {nsescreen}'):
        #Finding the top 30% performing stocks (Relative to the Nifty 50)
        for i in range(len(nifty_50)):
            #Downloading Historical Data
            df = yf.download(nifty_50[i], period=period)
            name = nifty_50[i]
            df.to_csv(f'csv/NSE/{name}.csv')

            #Calculating Returns Relative to the Market (returns Multiple)
            df['Percent Change'] = df['Adj Close'].pct_change()
            stock_return = (df['Percent Change'] + 1).cumprod()[-1]

            returns_multiple = round((stock_return / index_return), 2)
            returns_multiples.extend([returns_multiple])

    #Creating DataFrame of only Top 30%
    Relative_strength_df = pd.DataFrame(list(zip(nifty_50, returns_multiples)), columns=['Stocks', 'Returns_Multiple'])
    Relative_strength_df['RS_Rating'] = Relative_strength_df.Returns_Multiple.rank(pct=True) * 100
    Relative_strength_df = Relative_strength_df[Relative_strength_df.RS_Rating >= Relative_strength_df.RS_Rating.quantile(.70)]

    #Checking Conditions of the top 30% of Stocks in the List
    rs_stocks = Relative_strength_df['Stocks']
    for stock in rs_stocks:
        try:
            df = pd.read_csv(f'csv/NSE/{stock}.csv', index_col=0)
            SMA = [50, 100, 200]
            for x in SMA:
                df["EMA_"+str(x)] = round(df['Adj Close'].ewm(span=x).mean(), 2)

            #Storing Required Values
            current_close = df['Adj Close'][-1]
            moving_average_50 = df[f'EMA_{SMA[0]}'][-1]
            moving_average_150 = df[f"EMA_{SMA[1]}"][-1]
            moving_average_200 = df[f"EMA_{SMA[2]}"][-1]
            low_of_52week = round(min(df["Low"][-260:]), 2)
            high_of_52week = round(max(df["High"][-260:]), 2)
            RS_Rating = round(
                Relative_strength_df[Relative_strength_df['Stocks'] == stock].RS_Rating.tolist()[0])
            try:
                moving_average_200_20 = df["EMA_200"][-20]
            except Exception:
                moving_average_200_20 = 0

            #Condition 1 : Current Price > 150 SMA > 200 SMA
            condition_1 = current_close > moving_average_150 > moving_average_200

            #Condition 2 : 150 SMA > 200 SMA
            condition_2 = moving_average_150 > moving_average_200

            # Condition 3: 200 SMA trending up for at least 1 month
            condition_3 = moving_average_200 > moving_average_200_20

            # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
            condition_4 = moving_average_50 > moving_average_150 > moving_average_200

            # Condition 5: Current Price > 50 SMA
            condition_5 = current_close > moving_average_50

            # Condition 6: Current Price is at least 30% above 52 week low
            condition_6 = current_close >= (1.3*low_of_52week)

            # Condition 7: Current Price is within 25% of 52 week high
            condition_7 = current_close >= (.75*high_of_52week)

            if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7):
                exportList = exportList.append({'Stock': stock, 'Relative Strength Rating': RS_Rating, 'Current Price': current_close, "50 Day EMA": moving_average_50,
                    "150 Day EMA": moving_average_150, "200 Day EMA": moving_average_200, "52 Week Low": low_of_52week, "52 week High": high_of_52week}, ignore_index=True)
        except Exception as e:
            print(e)
            print(f'Could Not gather data on {stock}')

    if nsescreen == 'NIFTY 50':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_50.csv')
    elif nsescreen == 'NIFTY 100':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_100.csv')
    elif nsescreen == 'NIFTY 200':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_200.csv')
    elif nsescreen == 'NIFTY 500':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_500.csv')
    elif nsescreen == 'NIFTY Midcap 50':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_midcap50.csv')
    elif nsescreen == 'NIFTY Midcap 100':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_midcap100.csv')
    elif nsescreen == 'NIFTY Smallcap 50':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_smallcap50.csv')
    elif nsescreen == 'NIFTY Smallcap 250':
        exportList = exportList.sort_values(by='Relative Strength Rating', ascending=False)
        exportList.to_csv('screen/NSE/screen_smallcap250.csv')
    else:
        print('Error in code!')

screen = st.sidebar.checkbox('Screen NSE stocks from NIFTY Index')
if screen:
    nsescreen = st.sidebar.selectbox('Select NSE Index for Screening of Stocks', ['NIFTY 50', 'NIFTY 100', 'NIFTY 200', 'NIFTY 500', 'NIFTY Midcap 50', 'NIFTY Midcap 100', 'NIFTY Smallcap 50', 'NIFTY Smallcap 250'])
    if 'NIFTY 50' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_50.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_50.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY 100' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_100.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_100.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY 200' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_200.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_200.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY 500' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_500.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_500.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY Midcap 50' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_midcap50.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_midcap50.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY Midcap 100' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_midcap100.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_midcap100.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY Smallcap 50' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_smallcap50.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_smallcap50.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)
    if 'NIFTY Smallcap 250' in nsescreen:
        try:
            data = pd.read_csv('screen/NSE/screen_smallcap250.csv')
        except:
            screen_stocks(nsescreen=nsescreen, period=period)
        data = pd.read_csv('screen/NSE/screen_smallcap250.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.set_index('Stock', inplace=True)
        st.header(f'Screened Stocks from {nsescreen}')
        st.dataframe(data)

# Portfolio Section
def get_portfolio(portfolio, value, period):
    def get_key(val):
        for key, value in dicts.items():
            if val == value:
                return key
        return "key doesn't exist"

    if portfolio == 'NIFTY 50':
        try:
            df = pd.read_csv('portfolio/port_50.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_nifty50list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_50.csv')
            df = pd.read_csv('portfolio/port_50.csv', index_col=0)
    elif portfolio == 'NIFTY 100':
        try:
            df = pd.read_csv('portfolio/port_100.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_nifty100list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_100.csv')
            df = pd.read_csv('portfolio/port_100.csv', index_col=0)
    elif portfolio == 'NIFTY 200':
        try:
            df = pd.read_csv('portfolio/port_200.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_nifty200list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_200.csv')
            df = pd.read_csv('portfolio/port_200.csv', index_col=0)
    elif portfolio == 'NIFTY Midcap 50':
        try:
            df = pd.read_csv('portfolio/port_mid_50.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_niftymidcap50list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_midcap_50.csv')
            df = pd.read_csv('portfolio/port_midcap_50.csv', index_col=0)
    elif portfolio == 'NIFTY Midcap 100':
        try:
            df = pd.read_csv('portfolio/port_midcap_100.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_niftymidcap100list.csv')
            nifty_50 = []
            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_midcap_100.csv')
            df = pd.read_csv('portfolio/port_midcap_100.csv', index_col=0)
    elif portfolio == 'NIFTY Smallcap 50':
        try:
            df = pd.read_csv('portfolio/port_smallcap_50.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_niftysmallcap50list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_smallcap_50.csv')
            df = pd.read_csv('portfolio/port_smallcap_50.csv', index_col=0)
    elif portfolio == 'NIFTY Smallcap 250':
        try:
            df = pd.read_csv('portfolio/port_smallcap_250.csv', index_col=0)
        except:
            lists = pd.read_csv('dataset/ind_niftysmallcap250list.csv')
            nifty_50 = []
            for i in range(len(lists)):
                nifty_50.append(lists['Symbol'][i]+'.NS')

            df = pd.DataFrame()
            for i in range(len(nifty_50)):
                name = str(nifty_50[i])
                if i == 1:
                    dfs = yf.download(name, period=period)
                    dfs.reset_index(inplace=True)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs
                else:
                    dfs = yf.download(name, period=period)
                    dfs = dfs['Close'].to_list()
                    df[name] = dfs

            df.to_csv('portfolio/port_smallcap_250.csv')
            df = pd.read_csv('portfolio/port_smallcap_250.csv', index_col=0)
    else:
        st.write('Error in Code!')

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    #Optimize for the maximal Sharpe Ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

    cleaned_weigths = ef.clean_weights()
    expanlret, anlvol, sharpe = ef.portfolio_performance()
    expanlret = round(expanlret * 100, 2)
    anlvol = round(anlvol * 100, 2)

    portfolio_val = value
    latest_prices = get_latest_prices(df)
    weights = cleaned_weigths
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
    allocation, leftover = da.lp_portfolio()

    port = pd.DataFrame()
    tick = list(allocation.keys())
    share = list(allocation.values())
    port['Ticker'] = tick
    port['No of Shares'] = share
    comp_name = []
    for i in range(len(port)):
        val = port['Ticker'][i]
        cmp_name = get_key(val)
        comp_name.append(cmp_name)
    port['Company Name'] = comp_name

    cur_price = []
    for i in range(len(port)):
        df = yf.download(port['Ticker'][i], period='1d')
        price = df.iloc[0]['Close']
        price = round(price, 2)
        cur_price.append(price)
    port['Current Price'] = cur_price

    shares_val = port['No of Shares'].to_list()
    cur_price = port['Current Price'].to_list()

    vals = []
    for i in range(len(port)):
        val = shares_val[i] * cur_price[i]
        val = round(val, 2)
        vals.append(val)
    port['Invested Value'] = vals
    col_names = ['Company Name', 'Ticker', 'No of Shares', 'Current Price', 'Invested Value']
    port = port.reindex(columns=col_names)

    col1, col2, col3 = st.columns([2, 6, 2])
    col2.header('Portfolio for Nifty 50 Index')
    col2.dataframe(port)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(port)
    dwn_label = f'Download Your Portfolio'
    st.sidebar.download_button(label=dwn_label, data=csv, file_name='Portfolio.csv', mime='text/csv')

    st.write("==============================================================================================================================================================================")
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=port['Company Name'], values=port['No of Shares'], name="Portfolio Distribution",
                        hole=.45, hoverinfo="label+percent+value", textinfo='label+value+percent', textposition='inside'),  1, 1)
    fig.add_trace(go.Pie(labels=port['Company Name'], values=port['Invested Value'], name="Portfolio Invested Value",
                        hole=.45, hoverinfo="label+value+percent", textinfo='label+value+percent', textposition='inside'),  1, 2)
    fig.update_layout(title_text="Portofolio of Nifty 50 Stocks", height=850, legend=dict(orientation="h"),
        annotations=[dict(text='Number of shares', x=0.16, y=0.5, font_size=22, showarrow=False),
                    dict(text='Invested Value in each Share', x=0.86, y=0.5, font_size=18, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)
    st.write("==============================================================================================================================================================================")
    st.subheader("Expected annual return: {} % | | \tAnnual volatility: {} % | | \tFunds Remaining : RS {} out of RS {}".format(
        expanlret, anlvol, round(leftover, 2), math.floor(portfolio_val)))

port = st.sidebar.checkbox('Portfolio Creator')
if port:
    with st.expander("Portfolio Creator"):
        portfolio = st.selectbox('Select NSE Index for Creating of Protfolio', ['NIFTY 50', 'NIFTY 100', 'NIFTY 200', 'NIFTY Midcap 50', 'NIFTY Midcap 100', 'NIFTY Smallcap 50', 'NIFTY Smallcap 250'])
        value = st.number_input('Enter Portfolio Value in Rs.')
        if value == 0 or 0.00:
            pass
        else:
            with st.spinner('Creating your Portfolio...'):
                get_portfolio(portfolio=portfolio, value=value, period=period)

def LongShort(intraday):
    short = []
    long = []
    if intraday == 'NIFTY 50':
        data = pd.read_csv('dataset/ind_nifty50list.csv')
        df = data['Symbol']+'.NS'
        code = 1
    elif intraday == 'NIFTY 100':
        data = pd.read_csv('dataset/ind_nifty100list.csv')
        df = data['Symbol']+'.NS'
        code = 2
    elif intraday == 'NIFTY 200':
        data = pd.read_csv('dataset/ind_nifty200list.csv')
        df = data['Symbol']+'.NS'
        code = 3
    elif intraday == 'NIFTY 500':
        data = pd.read_csv('dataset/ind_nifty200list.csv')
        df = data['Symbol']+'.NS'
        code = 4
    elif intraday == 'NIFTY Midcap 50':
        data = pd.read_csv('dataset/ind_niftymidcap50list.csv')
        df = data['Symbol']+'.NS'
        code = 5
    elif intraday == 'NIFTY Midcap 100':
        data = pd.read_csv('dataset/ind_niftymidcap100list.csv')
        df = data['Symbol']+'.NS'
        code = 6
    elif intraday == 'NIFTY Smallcap 50':
        data = pd.read_csv('dataset/ind_niftysmallcap50list.csv')
        df = data['Symbol']+'.NS'
        code = 6
    elif intraday == 'NIFTY Smallcap 250':
        data = pd.read_csv('dataset/ind_niftysmallcap250list.csv')
        df = data['Symbol']+'.NS'
        code = 7
    else:
        pass

    with st.spinner('Getting your Stock to your Long/Short Intraday Positions...'):
        for i in range(len(df)):
            stock = df[i]
            data = yf.download(stock, period='1d')
            if data['Open'][0] == data['High'][0]:
                short.append(stock)
            elif data['Open'][0] == data['Low'][0]:
                long.append(stock)
            else:
                pass

    intra = pd.DataFrame()
    intra['Stocks to Short'] = pd.Series(short)
    intra['Stocks to Long'] = pd.Series(long)

    if code == 1:
        intra.to_csv('intraday/intra_50.csv')
    elif code == 2:
        intra.to_csv('intraday/intra_100.csv')
    elif code == 3:
        intra.to_csv('intraday/intra_200.csv')
    elif code == 4:
        intra.to_csv('intraday/intra_500.csv')
    elif code == 5:
        intra.to_csv('intraday/intramid_50.csv')
    elif code == 6:
        intra.to_csv('intraday/intramid_250.csv')
    elif code == 7:
        intra.to_csv('intraday/intrasmall_50.csv')
    elif code == 8:
        intra.to_csv('intraday/intrasmall_250.csv')
    else:
        pass
    st.dataframe(intra)

longshort = st.sidebar.checkbox('Find Stock for Short/Long Position for Intraday')
if longshort:
    intraday = st.selectbox('Select Index for Short/Long Position for Intraday', ['NIFTY 50', 'NIFTY 100', 'NIFTY 200', 'NIFTY 500', 'NIFTY Midcap 50', 'NIFTY Midcap 100', 'NIFTY Smallcap 50', 'NIFTY Smallcap 250'])
    if 'NIFTY 50' in intraday:
        try:
            intraday = pd.read_csv('intraday/intra_50.csv')
            st.dataframe(intraday)
        except:
            LongShort(intraday)
    if 'NIFTY 100' in intraday:
        try:
            intraday = pd.read_csv('intraday/intra_100.csv')
            st.dataframe(intraday)
        except:
            LongShort(intraday=intraday)
    if 'NIFTY 200' in intraday:
        try:
            intraday = pd.read_csv('intraday/intra_200.csv')
            st.dataframe(intraday)
        except:
            screen_stocks(intraday=intraday)
    if 'NIFTY 500' in intraday:
        try:
            intraday = pd.read_csv('intraday/intra_500.csv')
            st.dataframe(intraday)
        except:
            screen_stocks(intraday=intraday)
    if 'NIFTY Midcap 50' in intraday:
        try:
            data = pd.read_csv('intraday/intramid_50.csv')
            st.dataframe(intraday)
        except:
            screen_stocks(intraday=intraday, period=period)
    if 'NIFTY Midcap 100' in intraday:
        try:
            data = pd.read_csv('intraday/intramid_100.csv')
            st.dataframe(intraday)
        except:
            screen_stocks(intraday=intraday)
    if 'NIFTY Smallcap 50' in intraday:
        try:
            data = pd.read_csv('intraday/intrasmall_50.csv')
            st.dataframe(intraday)
        except:
            screen_stocks(intraday=intraday)
    if 'NIFTY Smallcap 250' in intraday:
        try:
            intraday = pd.read_csv('intraday/intrasmall_250.csv')
            st.dataframe(intraday)
        except:
            LongShort(intraday)

showdata = st.sidebar.checkbox(f'Show Dataset for {StockName}')
if showdata:
    st.dataframe(df)

# Momentum Indicatorss
def macd():
    with st.expander('MACD INFORMATION'):
            st.header("What Is Moving Average Convergence Divergence(MACD)?")
            st.subheader("Moving average convergence divergence(MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.")
            st.write("""
                    The MACD is calculated by subtracting the 26-period exponential moving average(EMA) from the 12-period EMA.
                    The result of that calculation is the MACD line. A nine-day EMA of the MACD called the signal line.
                    Signal Line is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.
                    Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.
                    Moving average convergence divergence(MACD) indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.""")
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                    1) Moving average convergence divergence(MACD) is calculated by subtracting the 26-period exponential moving average(EMA) from the 12-period EMA.\n
                    2) MACD triggers technical signals when it crosses above(to buy) or below(to sell) its signal line.\n
                    3) The speed of crossovers is also taken as a signal of a market is overbought or oversold.\n
                    4) MACD helps investors understand whether the bullish or bearish movement in the price is strengthening or weakening.""")
            st.subheader("MACD vs. Relative Strength")
            st.write("""
                    The relative strength indicator (RSI) aims to signal whether a market is considered to be overbought or oversold in relation to recent price levels.
                    The RSI is an oscillator that calculates average price gains and losses over a given period of time. The default time period is 14 periods with values bounded from 0 to 100.
                    MACD measures the relationship between two EMAs, while the RSI measures price change in relation to recent price highs and lows.
                    These two indicators are often used together to provide analysts a more complete technical picture of a market.
                    These indicators both measure momentum in a market, but, because they measure different factors, they sometimes give contrary indications.
                    For example, the RSI may show a reading above 70 for a sustained period of time, indicating a market is overextended to the buy-side in relation to recent prices, while the MACD indicates the market is still increasing in buying momentum. 
                    Either indicator may signal an upcoming trend change by showing divergence from price (price continues higher while the indicator turns lower, or vice versa).
            """)
            st.subheader("Limitations of MACD")
            st.write("""
                    One of the main problems with divergence is that it can often signal a possible reversal but then no actual reversal actually happens—it produces a false positive.
                    The other problem is that divergence doesn't forecast all reversals. In other words, it predicts too many reversals that don't occur and not enough real price reversals.
                    False positive" divergence often occurs when the price of an asset moves sideways, such as in a range or triangle pattern following a trend.
                    A slowdown in the momentum—sideways movement or slow trending movement—of the price will cause the MACD to pull away from its prior extremes and gravitate toward the zero lines even in the absence of a true reversal.
            """)

    # ## MACD (Moving Average Convergence Divergence) MACD
    df['MACD'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd()
    df['Hist'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_diff()
    df['Signal'] = ta.trend.MACD( df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_signal()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2], vertical_spacing=0.2, subplot_titles=['Price Chart', 'MACD'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=2), name='MACD'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#000000', width=2), name='Signal'), row=2, col=1)
    colors = np.where(df['Hist'] < 0, 'red', 'green')
    fig.append_trace(go.Bar(x=df.index,  y=df['Hist'], name='histogram', marker_color=colors), row=2, col=1)
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.10,
        title='Moving Average Convergence Divergence(MACD) Indicator', height=1000)
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)

def roc():
    with st.container():
        with st.expander('Rate of Change INFO'):
            st.header("What is the Price Rate Of Change (ROC) Indicator?")
            st.subheader("The Price Rate of Change (ROC) is a momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago.")
            st.write("""
                The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if price changes are to the upside, and moving into negative territory if price changes are to the downside.
                The indicator can be used to spot divergences, overbought and oversold conditions, and centerline crossovers.
                Like most momentum oscillators, the ROC appears on a chart in a separate window below the price chart. The ROC is plotted against a zero line that differentiates positive and negative values.
                Positive values indicate upward buying pressure or momentum, while negative values below zero indicate selling pressure or downward momentum.
                    """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) The Price Rate of Change (ROC) oscillator is and unbounded momentum indicator used in technical analysis set against a zero-level midpoint.\n
                2) A rising ROC above zero typically confirms an uptrend while a falling ROC below zero indicates a downtrend.\n
                3) When the price is consolidating, the ROC will hover near zero. In this case, it is important traders watch the overall price trend since the ROC will provide little insight except for confirming the consolidation.
                    """)
            st.subheader("What Does the Price Rate of Change Indicator Tell You?")
            st.write(""" 
                Increasing values in either direction, positive or negative, indicate increasing momentum, and moves back toward zero indicate waning momentum.
                Zero-line crossovers can be used to signal trend changes. Depending on the n value used these signal may come early in a trend change (small n value) or very late in a trend change (larger n value).
                ROC is prone to whipsaws, especially around the zero line. Therefore, this signal is generally not used for trading purposes, but rather to simply alert traders that a trend change may be underway.
                Overbought and oversold levels are also used. These levels are not fixed, but will vary by the asset being traded. Traders look to see what ROC values resulted in price reversals in the past. Often traders will find both positive and negative values where the price reversed with some regularity. 
                When the ROC reaches these extreme readings again, traders will be on high alert and watch for the price to start reversing to confirm the ROC signal. With the ROC signal in place, and the price reversing to confirm the ROC signal, a trade may be considered.
                ROC is also commonly used as a divergence indicator that signals a possible upcoming trend change. Divergence occurs when the price of a stock or another asset moves in one direction while its ROC moves in the opposite direction.
                For example, if a stock's price is rising over a period of time while the ROC is progressively moving lower, then the ROC is indicating bearish divergence from price, which signals a possible trend change to the downside. The same concept applies if the price is moving down and ROC is moving higher.
                This could signal a price move to the upside. Divergence is a notoriously poor timing signal since a divergence can last a long time and won't always result in a price reversal.
                """)
            st.subheader("Limitation of Using Rate of Change Indicator(ROC)")
            st.write("""
                One potential problem with using the ROC indicator is that its calculation gives equal weight to the most recent price and the price from n periods ago, despite the fact that some technical analysts consider more recent price action to be of more importance in determining likely future price movement.
                The indicator is also prone to whipsaws, especially around the zero line. This is because when the price consolidates the price changes shrink, moving the indicator toward zero. Such times can result in multiple false signals for trend trades, but does help confirm the price consolidation.
                While the indicator can be used for divergence signals, the signals often occur far too early. When the ROC starts to diverge, the price can still run in the trending direction for some time. Therefore, divergence should not be acted on as a trade signal, but could be used to help confirm a trade if other reversal signals are present from other indicators and analysis methods.
            """)

    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=12, fillna=True).roc()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'Rate of Change(ROC) Indicator'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['ROC'], line=dict(color='goldenrod', width=2), name='ROC'), row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2, line_dash='dash')
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Rate of Change(ROC) Indicator',
        height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def rsi():
    with st.container():
        with st.expander('Relative Strength Index (RSI) INFO'):
            st.header("What Is the Relative Strength Index (RSI)?")
            st.subheader("The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.")
            st.write("""
                The RSI is displayed as an oscillator (a line graph that moves between two extremes) and can have a reading from 0 to 100. The indicator was originally developed by J. Welles Wilder Jr. and introduced in his seminal 1978 book, “New Concepts in Technical Trading Systems.”
                Traditional interpretation and usage of the RSI are that values of 70 or above indicate that a security is becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price.
                An RSI reading of 30 or below indicates an oversold or undervalued condition.
                    """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) The relative strength index (RSI) is a popular momentum oscillator developed in 1978.\n
                2) The RSI provides technical traders with signals about bullish and bearish price momentum, and it is often plotted beneath the graph of an asset’s price.\n
                3) An asset is usually considered overbought when the RSI is above 70% and oversold when it is below 30%.
                """)
            st.subheader("What Does the RSI Tell You?")
            st.write("""
                The primary trend of the stock or asset is an important tool in making sure the indicator’s readings are properly understood. For example, well-known market technician Constance Brown, CMT, has promoted the idea that an oversold reading on the RSI in an uptrend is likely much higher than 30% and that an overbought reading on the RSI during a downtrend is much lower than the 70% level.1
                As you can see in the following chart, during a downtrend, the RSI would peak near the 50% level rather than 70%, which could be used by investors to more reliably signal bearish conditions.
                Many investors will apply a horizontal trendline between 30% and 70% levels when a strong trend is in place to better identify extremes. Modifying overbought or oversold levels when the price of a stock or asset is in a long-term horizontal channel is usually unnecessary.
                A related concept to using overbought or oversold levels appropriate to the trend is to focus on trade signals and techniques that conform to the trend. 
                In other words, using bullish signals when the price is in a bullish trend and bearish signals when a stock is in a bearish trend will help to avoid the many false alarms that the RSI can generate.
                """)
            st.subheader("Limitation of Using Relative Strength Index (RSI)")
            st.write("""
                The RSI compares bullish and bearish price momentum and displays the results in an oscillator that can be placed beneath a price chart. Like most technical indicators, its signals are most reliable when they conform to the long-term trend.
                True reversal signals are rare and can be difficult to separate from false alarms. A false positive, for example, would be a bullish crossover followed by a sudden decline in a stock. A false negative would be a situation where there is a bearish crossover, yet the stock suddenly accelerated upward.
                Since the indicator displays momentum, it can stay overbought or oversold for a long time when an asset has significant momentum in either direction. Therefore, the RSI is most useful in an oscillating market where the asset price is alternating between bullish and bearish movements.
                """)

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True).rsi()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'Relative Strength Index (RSI)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='grey', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=30, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Relative Strength Index (RSI)',
        height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def stochastic():
    with st.container():
        with st.expander('Stochastic Oscillator) INFO'):
            st.header("What Is a Stochastic Oscillator?")
            st.subheader("A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.")
            st.write("""
                The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result. It is used to generate overbought and oversold trading signals, utilizing a 0–100 bounded range of values.
                    """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) A stochastic oscillator is a popular technical indicator for generating overbought and oversold signals.\n
                2) It is a popular momentum indicator, first developed in the 1950s.\n
                3) Stochastic oscillators tend to vary around some mean price level, since they rely on an asset's price history.
                """)
            st.subheader("What Does the Stochastics Oscillator Tell You?")
            st.write("""
                The stochastic oscillator is range-bound, meaning it is always between 0 and 100. This makes it a useful indicator of overbought and oversold conditions. Traditionally, readings over 80 are considered in the overbought range, and readings under 20 are considered oversold.
                However, these are not always indicative of impending reversal; very strong trends can maintain overbought or oversold conditions for an extended period. Instead, traders should look to changes in the stochastic oscillator for clues about future trend shifts.
                Stochastic oscillator charting generally consists of two lines: one reflecting the actual value of the oscillator for each session, and one reflecting its three-day simple moving average. Because price is thought to follow momentum, the intersection of these two lines is considered to be a signal that a reversal may be in the works, as it indicates a large shift in momentum from day to day.
                Divergence between the stochastic oscillator and trending price action is also seen as an important reversal signal. For example, when a bearish trend reaches a new lower low, but the oscillator prints a higher low, it may be an indicator that bears are exhausting their momentum and a bullish reversal is brewing.
                """)
            st.subheader("Limitation of Using Stochastic Oscillator")
            st.write("""
            The primary limitation of the stochastic oscillator is that it has been known to produce false signals. This is when a trading signal is generated by the indicator, yet the price does not actually follow through, which can end up as a losing trade.
            During volatile market conditions, this can happen quite regularly. One way to help with this is to take the price trend as a filter, where signals are only taken if they are in the same direction as the trend.
                """)

    df['%K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3, fillna=True).stoch()
    df['%D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3, fillna=True).stoch_signal()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2], vertical_spacing=0.2, subplot_titles=['Price Chart', 'Stochastics Oscillator'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='orange', width=2), name='Fast'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='#000000', width=2), name='Slow'), row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Stochastics Oscillator ', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

# Volume Indicators
def mfi():
    with st.container():
        with st.expander('Money Flow Index(MFI) INFO'):
            st.header("What Is the Money Flow Index (MFI)?")
            st.subheader("The Money Flow Index (MFI) is a technical oscillator that uses price and volume data for identifying overbought or oversold signals in an asset. It can also be used to spot divergences which warn of a trend change in price. The oscillator moves between 0 and 100.")
            st.write("""
                Unlike conventional oscillators such as the Relative Strength Index (RSI), the Money Flow Index incorporates both price and volume data, as opposed to just price. For this reason, some analysts call MFI the volume-weighted RSI.
                """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) The Money Flow Index (MFI) is a technical indicator that generates overbought or oversold signals using both prices and volume data.\n
                2) An MFI reading above 80 is considered overbought and an MFI reading below 20 is considered oversold, although levels of 90 and 10 are also used as thresholds.\n
                3) A divergence between the indicator and price is noteworthy. For example, if the indicator is rising while the price is falling or flat, the price could start rising.
                """)
            st.subheader("What Does the Money Flow Index Tell You?")
            st.write("""
                One of the primary ways to use the Money Flow Index is when there is a divergence. A divergence is when the oscillator is moving in the opposite direction of price. This is a signal of a potential reversal in the prevailing price trend.
                For example, a very high Money Flow Index that begins to fall below a reading of 80 while the underlying security continues to climb is a price reversal signal to the downside. Conversely, a very low MFI reading that climbs above a reading of 20 while the underlying security continues to sell off is a price reversal signal to the upside.
                Traders also watch for larger divergences using multiple waves in the price and MFI. For example, a stock peaks at $10, pulls back to $8, and then rallies to $12. The price has made two successive highs, at $10 and $12. 
                If MFI makes a lower higher when the price reaches $12, the indicator is not confirming the new high. This could foreshadow a decline in price.
                The overbought and oversold levels are also used to signal possible trading opportunities. Moves below 10 and above 90 are rare. Traders watch for the MFI to move back above 10 to signal a long trade, and to drop below 90 to signal a short trade.
                Other moves out of overbought or oversold territory can also be useful. For example, when an asset is in an uptrend, a drop below 20 (or even 30) and then a rally back above it could indicate a pullback is over and the price uptrend is resuming.
                The same goes for a downtrend. A short-term rally could push the MFI up to 70 or 80, but when it drops back below that could be the time to enter a short trade in preparation for another drop.
                """)
            st.subheader("Limitation of Using Money Flow Index")
            st.write("""
            The MFI is capable of producing false signals. This is when the indicator does something that indicates a good trading opportunity is present, but then the price doesn't move as expected resulting in a losing trade. A divergence may not result in a price reversal, for instance.
            The indicator may also fail to warn of something important. For example, while a divergence may result in a price reversing some of the time, divergence won't be present for all price reversals. Because of this, it is recommended that traders use other forms of analysis and risk control and not rely exclusively on one indicator.
            """)
    df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Volume'], df['Close'], window=14, fillna=True).money_flow_index()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'Money Flow Index(MFI)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='grey', width=2), name='MFI'), row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True,
        title='Money Flow Index(MFI) Indicator', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def obv():
    with st.expander('On Balance Volume(OBV) INFO'):
        st.header("What is On-Balance Volume (OBV)?")
        st.subheader("On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price. Joseph Granville first developed the OBV metric in the 1963 book Granville's New Key to Stock Market Profits.")
        st.write("""
            Granville believed that volume was the key force behind markets and designed OBV to project when major moves in the markets would occur based on volume changes. In his book, he described the predictions generated by OBV as "a spring being wound tightly."
            He believed that when volume increases sharply without a significant change in the stock's price, the price will eventually jump upward or fall downward.
            """)
        st.subheader("KEY TAKEAWAYS")
        st.write("""
                1) On-balance volume (OBV) is a technical indicator of momentum, using volume changes to make price predictions.\n
                2) OBV shows crowd sentiment that can predict a bullish or bearish outcome.\n
                3) Comparing relative action between price bars and OBV generates more actionable signals than the green or red volume histograms commonly found at the bottom of price charts. \n
                """)
        st.subheader("What Does On-Balance Volume Tell You?")
        st.write("""
                The theory behind OBV is based on the distinction between smart money – namely, institutional investors – and less sophisticated retail investors. As mutual funds and pension funds begin to buy into an issue that retail investors are selling, volume may increase even as the price remains relatively level.
                Eventually, volume drives the price upward. At that point, larger investors begin to sell, and smaller investors begin buying.
                Despite being plotted on a price chart and measured numerically, the actual individual quantitative value of OBV is not relevant. The indicator itself is cumulative, while the time interval remains fixed by a dedicated starting point, meaning the real number value of OBV arbitrarily depends on the start date.
                Instead, traders and analysts look to the nature of OBV movements over time; the slope of the OBV line carries all of the weight of analysis.
                Analysts look to volume numbers on the OBV to track large, institutional investors. They treat divergences between volume and price as a synonym of the relationship between "smart money" and the disparate masses, hoping to showcase opportunities for buying against incorrect prevailing trends.
                For example, institutional money may drive up the price of an asset, then sell after other investors jump on the bandwagon.
                """)
        st.subheader("Limitations of using OBV")
        st.write("""
            One limitation of OBV is that it is a leading indicator, meaning that it may produce predictions, but there is little it can say about what has actually happened in terms of the signals it produces. Because of this, it is prone to produce false signals.
            It can therefore be balanced by lagging indicators. Add a moving average line to the OBV to look for OBV line breakouts; you can confirm a breakout in the price if the OBV indicator makes a concurrent breakout.
            Another note of caution in using the OBV is that a large spike in volume on a single day can throw off the indicator for quite a while.
            For instance, a surprise earnings announcement, being added or removed from an index, or massive institutional block trades can cause the indicator to spike or plummet, but the spike in volume may not be indicative of a trend.
            """)

    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume'],fillna=True).on_balance_volume()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'On Balance Volume(OBV)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='grey', width=2), name='OBV'), row=2, col=1)
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='On Balance Volume(OBV) Indicator', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def vwap():
    with st.expander('Volume-Weighted Average Price (VWAP) INFO'):
        st.header("What Is the Volume-Weighted Average Price (VWAP)?")
        st.write("""
                The volume-weighted average price (VWAP) is a trading benchmark used by traders that gives the average price a security has traded at throughout the day, based on both volume and price.
                VWAP is important because it provides traders with insight into both the trend and value of a security.
        """)
        st.subheader("KEY TAKEAWAYS")
        st.write("""
                1) The volume-weighted average price (VWAP) appears as a single line on intraday charts (1 minute, 15 minute, and so on), similar to how a moving average looks.\n
                2) Retail and professional traders may use the VWAP as part of their trading rules for determining intraday trends.\n
        """)
        st.subheader("What Does VWAP Volume Tell You?")
        st.write("""
                Large institutional buyers and mutual funds use the VWAP ratio to help move into or out of stocks with as small of a market impact as possible. Therefore, when possible, institutions will try to buy below the VWAP, or sell above it.
                This way their actions push the price back toward the average, instead of away from it.
                Traders may use VWAP as a trend confirmation tool, and build trading rules around it. For example, when the price is above VWAP, they may prefer to initiate long positions. When the price is below VWAP they may prefer to initiate short positions.
                """)
        st.subheader("Limitations of Using the Volume-Weighted Average Price")
        st.write("""
                VWAP is a single-day indicator and is restarted at the open of each new trading day. Attempting to create an average VWAP over many days could mean that the average becomes distorted from the true VWAP reading as described above.
                While some institutions may prefer to buy when the price of a security is below the VWAP, or sell when it is above, VWAP is not the only factor to consider. In strong uptrends, the price may continue to move higher for many days without dropping below the VWAP at all or only occasionally.
                Therefore, waiting for the price to fall below VWAP could mean a missed opportunity if prices are rising quickly. VWAP is based on historical values and does not inherently have predictive qualities or calculations.
                Because VWAP is anchored to the opening price range of the day, the indicator increases its lag as the day goes on. This can be seen in the way a 1-minute period VWAP calculation after 330 minutes (the length of a typical trading session) will often resemble a 390-minute moving average at the end of the trading day.
        """)

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14, fillna=True).volume_weighted_average_price()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'Volume Weighted Average Price(VWAP)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='grey', width=2), name='VWAP'), row=1, col=1)
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Volume Weighted Average Price Indicator',
    height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

# Volatility Indicators
def atr():
    with st.expander('Average True Range (ATR) INFO'):
            st.header("What Is the Average True Range (ATR)?")
            st.subheader("The average true range(ATR) is a technical analysis indicator, introduced by market technician J. Welles Wilder Jr. in his book New Concepts in Technical Trading Systems, that measures market volatility by decomposing the entire range of an asset price for that period.")
            st.write("""
                    The true range indicator is taken as the greatest of the following: current high less the current low; the absolute value of the current high less the previous close; and the absolute value of the current low less the previous close. The ATR is then a moving average, generally using 14 days, of the true ranges.
            """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                    1) The average true range (ATR) is a market volatility indicator used in technical analysis.\n
                    2) It is typically derived from the 14-day simple moving average of a series of true range indicators.\n
                    3) The ATR was originally developed for use in commodities markets but has since been applied to all types of securities.
            """)
            st.subheader("What Does the Average True Range (ATR) Tell You?")
            st.write("""
                Wilder originally developed the ATR for commodities, although the indicator can also be used for stocks and indices. Simply put, a stock experiencing a high level of volatility has a higher ATR, and a low volatility stock has a lower ATR.
                The ATR may be used by market technicians to enter and exit trades, and is a useful tool to add to a trading system. It was created to allow traders to more accurately measure the daily volatility of an asset by using simple calculations.
                The indicator does not indicate the price direction; rather it is used primarily to measure volatility caused by gaps and limit up or down moves. The ATR is fairly simple to calculate and only needs historical price data.
                The ATR is commonly used as an exit method that can be applied no matter how the entry decision is made. One popular technique is known as the "chandelier exit" and was developed by Chuck LeBeau. The chandelier exit places a trailing stop under the highest high the stock reached since you entered the trade.
                The distance between the highest high and the stop level is defined as some multiple times the ATR.
                For example, we can subtract three times the value of the ATR from the highest high since we entered the trade.
                """)
            st.subheader("Limitations of the Average True Range (ATR)")
            st.write("""
                There are two main limitations to using the ATR indicator. The first is that ATR is a subjective measure, meaning that it is open to interpretation. There is no single ATR value that will tell you with any certainty that a trend is about to reverse or not. 
                Instead, ATR readings should always be compared against earlier readings to get a feel of a trend's strength or weakness.
                Second, ATR only measures volatility and not the direction of an asset's price. This can sometimes result in mixed signals, particularly when markets are experiencing pivots or when trends are at turning points. 
                For instance, a sudden increase in the ATR following a large move counter to the prevailing trend may lead some traders to think the ATR is confirming the old trend; however, this may not actually be the case.
            """)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'Average True Range(ATR)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['ATR'], line=dict(color='grey', width=2), name='ATR'), row=2, col=1)
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Average True Range(ATR)',
        height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def bollinger():
    with st.container():
        with st.expander('Bollinger Bands INFO'):
            st.header("What Is a Bollinger Bands?")
            st.subheader("A Bollinger Band is a technical analysis tool defined by a set of trendlines plotted two standard deviations(positively and negatively) away from a simple moving average(SMA) of a security's price, but which can be adjusted to user preferences.")
            st.write("""
            Bollinger Bands were developed and copyrighted by famous technical trader John Bollinger, designed to discover opportunities that give investors a higher probability of properly identifying when an asset is oversold or overbought.
                """)
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) Bollinger Bands® are a technical analysis tool developed by John Bollinger for generating oversold or overbought signals.\n
                2) There are three lines that compose Bollinger Bands: A simple moving average (middle band) and an upper and lower band.\n
                3) The upper and lower bands are typically 2 standard deviations +/- from a 20-day simple moving average, but they can be modified.
                """)
            st.subheader("What Does Bollinger Bands Tell You?")
            st.write("""
                Bollinger Bands® are a highly popular technique. Many traders believe the closer the prices move to the upper band, the more overbought the market, and the closer the prices move to the lower band, the more oversold the market.
                John Bollinger has a set of 22 rules to follow when using the bands as a trading system.
                In the chart depicted below, Bollinger Bands® bracket the 20-day SMA of the stock with an upper and lower band along with the daily movements of the stock's price. 
                Because standard deviation is a measure of volatility, when the markets become more volatile the bands widen; during less volatile periods, the bands contract.
                """)
            st.subheader("Limitation of Using Bollinger Bands")
            st.write("""
           Bollinger Bands® are not a standalone trading system. They are simply one indicator designed to provide traders with information regarding price volatility. John Bollinger suggests using them with two or three other non-correlated indicators that provide more direct market signals.
           He believes it is crucial to use indicators based on different types of data. Some of his favored technical techniques are moving average divergence/convergence (MACD), on-balance volume, and relative strength index (RSI).
           Because they are computed from a simple moving average, they weigh older price data the same as the most recent, meaning that new information may be diluted by outdated data.
           Also, the use of 20-day SMA and 2 standard deviations is a bit arbitrary and may not work for everyone in every situation. Traders should adjust their SMA and standard deviation assumptions accordingly and monitor them. 
           """)

    df['Boll_H'] = ta.volatility.BollingerBands(df['Close'], window= 20, window_dev = 2, fillna = True).bollinger_hband()
    df['Boll_L'] = ta.volatility.BollingerBands(df['Close'], window= 20, window_dev = 2, fillna = True).bollinger_lband()
    df['Boll_M'] = ta.volatility.BollingerBands(df['Close'], window= 20, window_dev = 2, fillna = True).bollinger_mavg()

    fig = go.Figure(data=[go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
        go.Scatter(x=df.index, y=df['Boll_H'],line_color = 'gray', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.5),
        go.Scatter(x=df.index, y=df['Boll_M'],line_color = 'gray', line = {'dash': 'dash'},fill = 'tonexty', name = 'Middle Band', opacity = 0.5),
        go.Scatter(x=df.index, y=df['Boll_L'],line_color = 'gray', line = {'dash': 'dash'},fill = 'tonexty', name = 'Low Band', opacity = 0.5)])
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Bollinger Bands Indicator', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

#Trend Indicators
def sma():
    with st.container():
        with st.expander('Moving Average (MA) INFO'):
            st.header("What Is a Moving Average (MA)?")
            st.subheader("In statistics, a moving average is a calculation used to analyze data points by creating a series of averages of different subsets of the full data set. In finance, a moving average (MA) is a stock indicator that is commonly used in technical analysis.")
            st.write("""
                The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price.
                By calculating the moving average, the impacts of random, short-term fluctuations on the price of a stock over a specified time frame are mitigated.""")
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) A moving average (MA) is a stock indicator that is commonly used in technical analysis.\n
                2) The reason for calculating the moving average of a stock is to help smooth out the price data over a specified period of time by creating a constantly updated average price.\n
                3) A simple moving average (SMA) is a calculation that takes the arithmetic mean of a given set of prices over the specific number of days in the past; for example, over the previous 15, 30, 100, or 200 days.\n
                4) Exponential moving averages (EMA) is a weighted average that gives greater importance to the price of a stock in more recent days, making it an indicator that is more responsive to new information.
                """)
            st.header("Types of Moving Averages")
            st.subheader("Simple Moving Average")
            st.write("""
                The simplest form of a moving average, known as a simple moving average (SMA), is calculated by taking the arithmetic mean of a given set of values over a specified period of time. In other words, a set of numbers–or prices in the case of financial instruments–are added together and then divided by the number of prices in the set.
                """)
            st.subheader("Exponential Moving Average(EMA)")
            st.write("""The exponential moving average is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information. To calculate an EMA, you must first compute the simple moving average (SMA) over a particular time period.
            Next, you must calculate the multiplier for weighting the EMA (referred to as the "smoothing factor"), which typically follows the formula: [2/(selected time period + 1)]. So, for a 20-day moving average, the multiplier would be [2/(20+1)]= 0.0952. Then you use the smoothing factor combined with the previous EMA to arrive at the current value.
            The EMA thus gives a higher weighting to recent prices, while the SMA assigns an equal weighting to all values.""")
            st.subheader("What Does a Moving Average Indicate?")
            st.write("""
            A moving average is a statistic that captures the average change in a data series over time. In finance, moving averages are often used by technical analysts to keep track of prices trends for specific securities. An upward trend in a moving average might signify an upswing in the price or momentum of a security, while a downward trend would be seen as a sign of decline.
            Today, there is a wide variety of moving averages to choose from, ranging from simple measures to complex formulas that require a computer program to efficiently calculate.
            """)
            st.subheader("What Are Moving Averages Used for?")
            st.write("""Moving averages are widely used in technical analysis, a branch of investing that seeks to understand and profit from the price movement patterns of securities and indices. 
            Generally, technical analysts will use moving averages to detect whether a change in momentum is occurring for a security, such as if there is a sudden downward move in a security’s price.
            Other times, they will use moving averages to confirm their suspicions that a change might be underway. For example, if a company’s share price rises above its 200-day moving average, that might be taken as a bullish signal.
                """)

    df['SMA1'] = df['Close'].rolling(window=50, min_periods=0).mean()
    df['SMA2'] = df['Close'].rolling(window=200, min_periods=0).mean()

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA1'], name='SMA-20', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA2'], name='SMA-200', line=dict(color='grey', width=1)))
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.08,
        title='Simple Moving Average(SMA)', height=850)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

def ema():
    with st.container():
        with st.expander('Moving Average (MA) INFO'):
            st.header("What Is a Moving Average (MA)?")
            st.subheader("In statistics, a moving average is a calculation used to analyze data points by creating a series of averages of different subsets of the full data set. In finance, a moving average (MA) is a stock indicator that is commonly used in technical analysis.")
            st.write("""
                The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price.
                By calculating the moving average, the impacts of random, short-term fluctuations on the price of a stock over a specified time frame are mitigated.""")
            st.subheader("KEY TAKEAWAYS")
            st.write("""
                1) A moving average (MA) is a stock indicator that is commonly used in technical analysis.\n
                2) The reason for calculating the moving average of a stock is to help smooth out the price data over a specified period of time by creating a constantly updated average price.\n
                3) A simple moving average (SMA) is a calculation that takes the arithmetic mean of a given set of prices over the specific number of days in the past; for example, over the previous 15, 30, 100, or 200 days.\n
                4) Exponential moving averages (EMA) is a weighted average that gives greater importance to the price of a stock in more recent days, making it an indicator that is more responsive to new information.
                """)
            st.header("Types of Moving Averages")
            st.subheader("Simple Moving Average")
            st.write("""
                The simplest form of a moving average, known as a simple moving average (SMA), is calculated by taking the arithmetic mean of a given set of values over a specified period of time. In other words, a set of numbers–or prices in the case of financial instruments–are added together and then divided by the number of prices in the set.
                """)
            st.subheader("Exponential Moving Average(EMA)")
            st.write("""The exponential moving average is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information. To calculate an EMA, you must first compute the simple moving average (SMA) over a particular time period.
            Next, you must calculate the multiplier for weighting the EMA (referred to as the "smoothing factor"), which typically follows the formula: [2/(selected time period + 1)]. So, for a 20-day moving average, the multiplier would be [2/(20+1)]= 0.0952. Then you use the smoothing factor combined with the previous EMA to arrive at the current value.
            The EMA thus gives a higher weighting to recent prices, while the SMA assigns an equal weighting to all values.""")
            st.subheader("What Does a Moving Average Indicate?")
            st.write("""
            A moving average is a statistic that captures the average change in a data series over time. In finance, moving averages are often used by technical analysts to keep track of prices trends for specific securities. An upward trend in a moving average might signify an upswing in the price or momentum of a security, while a downward trend would be seen as a sign of decline.
            Today, there is a wide variety of moving averages to choose from, ranging from simple measures to complex formulas that require a computer program to efficiently calculate.
            """)
            st.subheader("What Are Moving Averages Used for?")
            st.write("""Moving averages are widely used in technical analysis, a branch of investing that seeks to understand and profit from the price movement patterns of securities and indices. 
            Generally, technical analysts will use moving averages to detect whether a change in momentum is occurring for a security, such as if there is a sudden downward move in a security’s price.
            Other times, they will use moving averages to confirm their suspicions that a change might be underway. For example, if a company’s share price rises above its 200-day moving average, that might be taken as a bullish signal.
                """)

    df['EMA_50'] = df['Close'].ewm(span=50, min_periods=0).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, min_periods=0).mean()

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'],high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA-50', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA-200', line=dict(color='grey', width=1)))
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.08,
        title='Exopnential Moving Average(EMA)', height=850)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

pattern = st.sidebar.multiselect("Select Candle Stick Pattern", ['Bullish Candle Stick Patterns', 'Bearish Candle Stick Patterns'])
if 'Bullish Candle Stick Patterns' in pattern:
    def bull():
        file = open("bull.py", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        return dictionary
    bull = bull()
    bull_candleStick = st.sidebar.selectbox('Select Bullish Candle Stick Pattern', bull.keys())
    bull_stick = bull.get(bull_candleStick)

    def bull_plot(bull_stick):
        new_df = pd.DataFrame()
        new_pattern = eval("talib.{}".format(bull_stick))
        new_df[f'bull_stick'] = new_pattern(
            df['Open'], df['High'], df['Low'], df['Close'])
        bear = []
        for i in range(len(df)):
            if new_df[f'bull_stick'][i] > 0:
                bear.append(df['Open'][i])
            else:
                bear.append(np.nan)
        df['Bull'] = bear
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Data')])
        fig.add_trace(go.Scatter(x=df.index, y=df['Bull'], marker_color='green', mode='markers+text', marker=dict(symbol='star-triangle-up', size=10), name="Bullish {}".format(bull_candleStick)))
        layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, height=800, legend=dict(orientation="h"))
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    bull_plot(bull_stick)

if 'Bearish Candle Stick Patterns' in pattern:
    def bear():
        file = open("bear.py", "r")
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        file.close()
        return dictionary
    bear = bear()
    bear_candleStick = st.sidebar.selectbox('Select Bearish Candle Stick Pattern', bear.keys())
    bear_stick = bear.get(bear_candleStick)

    def bear_plot(bear_stick):
        new_df = pd.DataFrame()
        new_pattern = eval("talib.{}".format(bear_stick))
        new_df[f'bear_stick'] = new_pattern(df['Open'], df['High'], df['Low'], df['Close'])
        bear = []
        for i in range(len(df)):
            if new_df[f'bear_stick'][i] < 0:
                bear.append(df['Open'][i])
            else:
                bear.append(np.nan)
        df['Bear'] = bear
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Data')])
        fig.add_trace(go.Scatter(x=df.index, y=df['Bear'], marker_color='red', mode='markers+text', marker=dict(symbol='star-triangle-up', size=10), name="Bearish {}".format(bear_candleStick)))
        layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, height=800, legend=dict(orientation="h"))
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    bear_plot(bear_stick)

indicators = st.sidebar.multiselect("Select the Indicator Type", ['Momentum Indicators', 'Trend Indicators', 'Volatility Indicators', 'Volume Indicators'])
if 'Momentum Indicators' in indicators:
    momentum = st.sidebar.selectbox('Select the Momentum Indicator', ['Moving Average Convergence Divergence (MACD)', 'Rate of Change (ROC)',\
         'Relative Strength Index (RSI)', 'Stochastic Oscillator'])
    if 'Moving Average Convergence Divergence (MACD)' in momentum:
        macd()
    if 'Rate of Change (ROC)' in momentum:
        roc()
    if 'Relative Strength Index (RSI)' in momentum:
        rsi()
    if 'Stochastic Oscillator' in momentum:
        stochastic()
if 'Volume Indicators' in indicators:
    volume = st.sidebar.selectbox('Select the Volume Indicator', ['Money Flow Index(MFI)', 'On Balance Volume(OBV)', 'Volume Weighted Average Price(VWAP)'])
    if 'Money Flow Index(MFI)' in volume:
        mfi()
    if 'On Balance Volume(OBV)' in volume:
        obv()
    if 'Volume Weighted Average Price(VWAP)' in volume:
        vwap()
if 'Volatility Indicators' in indicators:
    volatility = st.sidebar.selectbox('Select the Volatility Indicator', ['Average True Range(ATR)', 'Bollinger Bands'])
    if 'Average True Range(ATR)' in volatility:
        atr()
    if 'Bollinger Bands' in volatility:
        bollinger()
if 'Trend Indicators' in indicators:
    trend = st.sidebar.selectbox('Select the Trend Indicator', ['Exponential Moving Average(EMA)', 'Moving Average Convergence Divergence(MACD)',\
    'Relative Strength Index (RSI)', 'Simple Moving Average(SMA)', 'Stochastics Oscillator'])
    if 'Exponential Moving Average(EMA)' in trend:
        ema()
    if 'Moving Average Convergence Divergence(MACD)' in trend:
        macd()
    if 'Relative Strength Index (RSI)' in trend:
        rsi()
    if 'Simple Moving Average(SMA)' in trend:
        sma()
    if 'Stochastics Oscillator' in trend:
        stochastic()

#Fibonacci Retracement Trading Strategy
def fib_ret_strategy(df):
    #Calulate the Fibonacci Retracement Level
    max_price = df['Close'].max()
    min_price = df['Close'].min()

    difference = max_price - min_price
    first_level = max_price - difference * 0.236
    second_level = max_price - difference * 0.382
    third_level = max_price - difference * 0.5
    fourth_level = max_price - difference * 0.618

    #Calulate the MACD line and the Signal Line Indicators
    df['MACD'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd()
    df['Hist'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_diff()
    df['Signal'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_signal()

    #Create a function to get upper fib levels and lower fib levels of current Price
    def getlevels(price):
        if price >= first_level:
            return (max_price, first_level)
        elif price >= second_level:
            return (first_level, second_level)
        elif price >= third_level:
            return (second_level, third_level)
        elif price >= fourth_level:
            return (third_level, fourth_level)
        else:
            return (fourth_level, min_price)

    #Create a function for Trading Strategy
    def strategy(df):
        buy_list = []
        sell_list = []
        flag = 0
        last_buy_price = 0

        #Loop through the Data set
        for i in range(0, df.shape[0]):
            price = df["Close"][i]
            if i == 0:
                upper_lvl, lower_lvl = getlevels(price)
                buy_list.append(np.nan)
                sell_list.append(np.nan)
            elif price >= upper_lvl or price <= lower_lvl:

                if df['Signal'][i] > df['MACD'][i] and flag == 0:
                    last_buy_price = price
                    buy_list.append(price)
                    sell_list.append(np.nan)
                    # set the flag to 1 to signal that the share was bought
                    flag = 1
                elif df['Signal'][i] < df['MACD'][i] and flag == 1 and price >= last_buy_price:
                    buy_list.append(np.nan)
                    sell_list.append(price)
                    flag = 0
                else:
                    buy_list.append(np.nan)
                    sell_list.append(np.nan)
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)

            upper_lvl, lower_lvl = getlevels(price)

        return buy_list, sell_list

    # Create buy and sell coluns
    buy, sell = strategy(df)
    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    #Plot the Fibonacci Levels along with the close price and the MACD and Signal Price
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_width=[0.5, 1.3], subplot_titles=[f'Price Chart for {StockName}', 'MACD'])

    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',\
        marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',\
        marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)

    fig.add_hline(y=max_price, col=1, row=1, line_dash='dash', line_color="red", line_width=2)
    fig.add_hline(y=first_level, col=1, row=1, line_dash='dash', line_color="orange", line_width=2)
    fig.add_hline(y=second_level, col=1, row=1, line_dash='dash', line_color="#d1c819", line_width=2)
    fig.add_hline(y=third_level, col=1, row=1, line_dash='dash', line_color="darkgreen", line_width=2)
    fig.add_hline(y=fourth_level, col=1, row=1, line_dash='dash', line_color="darkblue", line_width=2)
    fig.add_hline(y=min_price, col=1, row=1, line_dash='dash', line_color="darkviolet", line_width=2)

    fig.append_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=2), name='MACD'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue', width=2), name='Signal'), row=2, col=1)
    colors = np.where(df['Hist'] < 0, 'red', 'green')
    fig.append_trace(go.Bar(x=df.index, y=df['Hist'], name='histogram', marker_color=colors), row=2, col=1)

    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.09,
        title='Fibonacci Retracement Trading Strategy',
        height=1000,
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

#Exp-Moving Average Strategy
def ema_strategy(df, ema):
    df = yf.download(stock, period=period)
    df['EMA'] = df['Close'].ewm(span=ema, min_periods=0).mean()

    def e_strategy(df):
        buy = []
        sell = []
        flag = 0
        buy_price = 0
        for i in range(0, len(df)):
            if df['EMA'][i] < df['Close'][i] and flag == 0:
                buy.append(df['Close'][i])
                sell.append(np.nan)
                buy_price = df['Close'][i]
                flag = 1
            elif df["EMA"][i] > df['Close'][i] and flag == 1 and buy_price < df['Close'][i]:
                sell.append(df['Close'][i])
                buy.append(np.nan)
                buy_price = 0
                flag = 0
            else:
                sell.append(np.nan)
                buy.append(np.nan)

        return (buy, sell)

    df['Buy_signal'] = e_strategy(df)[0]
    df['Sell_signal'] = e_strategy(df)[1]

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name=f'EMA-{ema}',line=dict(color='grey', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
    marker=dict(symbol='triangle-up', size=10), name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
    marker=dict(symbol='triangle-down', size=10), name='Sell'))

    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness = 0.08,
        title='Exopnential Moving Average(EMA) Trading Strategy', height=700)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

#Double Moving Average Strategy
def dsma_strategy(df, sma1, sma2):
    #Create the simple Moving Average with 30 day
    df = yf.download(stock, period=period)
    df['SMA_1'] = df['Close'].rolling(window=sma1, min_periods=0).mean()
    df['SMA_2'] = df['Close'].rolling(window=sma2, min_periods=0).mean()

    #Function for strategy

    def buy_sell(df):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1
        buy_price = 0
        for i in range(len(df)):
            if df['SMA_1'][i] > df["SMA_2"][i]:
                if flag != 1:
                    sigPriceBuy.append(df['Close'][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                    buy_price = df['Close'][i]
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            elif df['SMA_1'][i] < df['SMA_2'][i] and buy_price < df['Close'][i]:
                if flag != 0:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(df['Close'][i])
                    flag = 0
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        return (sigPriceBuy, sigPriceSell)

    buy_sell = buy_sell(df)
    df['Buy_signal'] = buy_sell[0]
    df['Sell_signal'] = buy_sell[1]

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_1'], name=f'SMA-{sma1}',line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_2'], name=f'SMA-{sma2}',line=dict(color='darkgrey', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
    marker=dict(symbol='triangle-up', size=10), name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
    marker=dict(symbol='triangle-down', size=10), name='Sell'))

    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness = 0.08,
        title='Double Simple Moving Average(SMA) Trading Strategy',
        height=700
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

#Double EMA Strategy
def dema_strategy(df, ema1, ema2):
    df = yf.download(stock, period=period)

    #Create the simple Moving Average with 30 day
    df['EMA_1'] = df['Close'].ewm(span=ema1, min_periods=0).mean()
    df['EMA_2'] = df['Close'].ewm(span=ema2, min_periods=0).mean()

    #Function for strategy
    def buy_sell(df):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1
        price = 0

        for i in range(len(df)):
            if df['EMA_1'][i] > df["EMA_2"][i]:
                if flag != 1:
                    sigPriceBuy.append(df['Close'][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            elif df['EMA_1'][i] < df['Close'][i] and price < df['Close'][i]:
                if flag != 0:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(df['Close'][i])
                    flag = 0
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        return (sigPriceBuy, sigPriceSell)

    buy_sell = buy_sell(df)
    df['Buy_signal'] = buy_sell[0]
    df['Sell_signal'] = buy_sell[1]

    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_1'], name=f'EMA-{ema1}',line=dict(color='grey', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_2'], name=f'EMA-{ema2}',line=dict(color='grey', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
    marker=dict(symbol='triangle-up', size=10), name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
    marker=dict(symbol='triangle-down', size=10), name='Sell'))

    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness = 0.08,
        title='Double-EMA Trading Strategy',
        height=700)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

# Stochastic EMA Strategy
def stoch_ema(df, stoch, ema, swin, swin1):
    df = yf.download(stock, period=period)
    df['%K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoch, smooth_window=swin, fillna=True).stoch()
    df['%D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoch, smooth_window=swin1, fillna=True).stoch_signal()
    df['EMA'] = df['Close'].ewm(span=ema, min_periods=0).mean()

    def strategy(df):
        buy_price = []
        sell_price = []
        flag = 0
        price = 0

        #Loop through the Data set
        for i in range(len(df)):
            if df['%K'][i] > 20 and df['%D'][i] > 20 and df['Close'][i] > df['EMA'][i]:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            elif df['%K'][i] < 80 and df['%D'][i] < 80 and df['Close'][i] < df['EMA'][i] and price < df['Close'][i]:
                if flag != -1 and flag !=0:
                    buy_price.append(np.nan)
                    sell_price.append(df['Close'][i])
                    flag = -1
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)

        return buy_price, sell_price

    # Create buy and sell coluns
    buy, sell = strategy(df)

    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2],
    subplot_titles=[f'Price Chart for {StockName}', 'Stochastics Oscillator'])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)

    fig.append_trace(go.Scatter(x=df.index, y=df['EMA'], marker_color='grey', name='EMA'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
    marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
    marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='orange', width=2), name='Fast'), row=2, col=1)

    fig.append_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='#000000', width=2), name='slow'), row=2, col=1)

    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')

    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Stochastic-EMA Trading Strategy', height=1000)
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)

#RSI-EMA Strategy
def rsi_ema(df, rsi, ema):
    df = yf.download(stock, period=period)

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi, fillna=True).rsi()
    df['EMA'] = df['Close'].ewm(span=ema, min_periods=0).mean()
    df.dropna(inplace=True)

    def strategy(df):
        buy_price = []
        sell_price = []
        flag = 0
        price = 0

        #Loop through the Data set
        for i in range(len(df)):
            if df['RSI'][i] > 50 and df['Close'][i] > df['EMA'][i]:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            elif df['RSI'][i] < 50 and df['Close'][i] < df['EMA'][i] and price < df['Close'][i]:
                if flag != -1 and flag !=0:
                    buy_price.append(np.nan)
                    sell_price.append(df['Close'][i])
                    flag = -1
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)

        return buy_price, sell_price

    # Create buy and sell coluns
    buy, sell = strategy(df)
    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5, 1.2],
        subplot_titles=[f'Price Chart for {StockName}', 'Relative Strenght Index(RSI)'])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)

    fig.append_trace(go.Scatter(x=df.index, y=df['EMA'], marker_color='grey', name='EMA'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
        marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
        marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)
    fig.append_trace(
        go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#000', width=2), name='RSI'), row=2, col=1)

    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    # Add overbought/oversold
    fig.add_hline(y=30, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')

    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        title='RSI-EMA Trading Strategy',
        height=1000,
    )
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)

#MACD Strategy
def macd_strat(df, fast, slow, signal):
    df = yf.download(stock, period=period)
    df['MACD'] = ta.trend.MACD(df['Close'], window_slow=slow, window_fast=fast, window_sign=signal, fillna=True).macd()
    df['Hist'] = ta.trend.MACD(df['Close'], window_slow=slow, window_fast=fast, window_sign=signal, fillna=True).macd_diff()
    df['Signal'] = ta.trend.MACD(df['Close'], window_slow=slow, window_fast=fast, window_sign=signal, fillna=True).macd_signal()
    df.dropna(inplace=True)

    def strategy(df):
        buy_price = []
        sell_price = []
        flag = 0
        price = 0

        for i in range(len(df)):
            if df['MACD'][i] > df['Signal'][i]:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            elif df['MACD'][i] < df['Signal'][i] and price < df['Close'][i]:
                if flag != -1 and flag != 0:
                    buy_price.append(np.nan)
                    sell_price.append(df['Close'][i])
                    flag = -1
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)

        return buy_price, sell_price

    buy, sell = strategy(df)
    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.9], subplot_titles=['Price Chart', 'MACD'])

    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
        marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
        marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=2), name='MACD'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#000000', width=2), name='Signal'), row=2, col=1)
    colors = np.where(df['Hist'] < 0, 'red', 'green')
    fig.append_trace(go.Bar(x=df.index,  y=df['Hist'], name='histogram', marker_color=colors), row=2, col=1)
    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        title='MACD Trading Strategy',
        height=1200,
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

# STOCH-MADC-RSI Strategy
def stoch_macd_rsi(stoc, macdf, macds, signal, rsi):
    df = yf.download(stock, period=period)
    df['%K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoc, smooth_window=3, fillna=True).stoch()
    df['%D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoc, smooth_window=3, fillna=True).stoch_signal()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi, fillna=True).rsi()
    df['MACD'] = ta.trend.MACD(df['Close'], window_slow=macds, window_fast=macdf, window_sign=signal, fillna=True).macd()
    df['Hist'] = ta.trend.MACD(df['Close'], window_slow=macds, window_fast=macdf, window_sign=signal, fillna=True).macd_diff()
    df['Signal'] = ta.trend.MACD(df['Close'], window_slow=macds, window_fast=macdf, window_sign=signal, fillna=True).macd_signal()
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df.dropna(inplace=True)

    def strategy(df):
        buy_price = []
        sell_price = []
        flag = 0
        price = 0

        for i in range(len(df)):
            if df['%K'][i] > 20 and df['%D'][i] > 20 and df['RSI'][i] > 30 and df['MACD'][i] > df['Signal'][i]:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            elif df['%K'][i] < 80 and df['%D'][i] < 80 and df['RSI'][i] < 70 and df['MACD'][i] < df['Signal'][i] and price < df['Close'][i]:
                if flag != -1 and flag != 0:
                    buy_price.append(np.nan)
                    sell_price.append(df['Close'][i])
                    flag = -1
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)

        return buy_price, sell_price

    # Create buy and sell coluns
    buy, sell = strategy(df)

    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    fig = make_subplots(rows=12, cols=1, shared_xaxes=True,
        specs=[[{"rowspan":5}], [None], [None], [None], [None], [None],[{"rowspan":2}], [None], [{"rowspan":2}], [None], [{"rowspan":2}], [None]],
    subplot_titles=[f'Price Chart for {StockName}', 'Stochastics', 'MACD', 'RSI Indicator'])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
    marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
    marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)

    #Stochastics
    fig.append_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='orange', width=2), name='Stoch-Fast'), row=7, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='#000000', width=2), name='Stoch-Slow'), row=7, col=1)

    #MACD
    fig.append_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange', width=2), name='MACD-Fast'), row=9, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='blue', width=2), name='MACD-Slow'), row=9, col=1)
    colors = np.where(df['Hist'] < 0, 'red', 'green')
    fig.append_trace(go.Bar(x=df.index, y=df['Hist'], name='histogram', marker_color=colors), row=9, col=1)

    #RSI
    fig.append_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#000000', width=2), name='RSI'), row=11, col=1)

    #Indicator borders
    fig.add_hline(y=0, col=1, row=7, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=7, line_color="#666", line_width=2)

    fig.add_hline(y=0, col=1, row=11, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=11, line_color="#666", line_width=2)

    # Add overbought/oversold
    #Stochastics
    fig.add_hline(y=20, col=1, row=7, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=7, line_color='#336699', line_width=2, line_dash='dash')
    #RSI
    fig.add_hline(y=30, col=1, row=11, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=11, line_color='#336699', line_width=2, line_dash='dash')

    layout = go.Layout(
        font_family='Monospace',
        titlefont_size=20,
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness = 0.08,
        title='Stochastic-MACD-RSI Trading Strategy',
        height=1000
    )
    fig.update_layout(layout)
    fig.update_xaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True)

#On Balance Volume(OBV) Strategy
def obv_strategy(df, obv_ema, ema):
    df = yf.download(stock, period=period)
    OBV = []
    OBV.append(0)
    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i-1]:
              OBV.append(OBV[-1] + df.Volume[i])
        elif df.Close[i] < df.Close[i-1]:
              OBV.append( OBV[-1] - df.Volume[i])
        else:
              OBV.append(OBV[-1])

    df['OBV'] = OBV
    df['OBV_EMA'] = df['OBV'].ewm(span=obv_ema, min_periods=0).mean()
    df['EMA'] = df['Close'].ewm(span=ema, min_periods=0).mean()

    def buy_sell(signal, col1, col2):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1
        price = 0
        for i in range(0,len(signal)):
            if signal[col1][i] > signal[col2][i] and flag != 1:
                sigPriceBuy.append(signal['Close'][i])
                sigPriceSell.append(np.nan)
                flag = 1
                price = df['Close'][i]
            elif signal[col1][i] < signal[col2][i] and flag != 0 and price < df['Close'][i]:
                sigPriceSell.append(signal['Close'][i])
                sigPriceBuy.append(np.nan)
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)

        return (sigPriceBuy, sigPriceSell)

    x = buy_sell(df, 'OBV','OBV_EMA' )
    df['Buy_signal'] = x[0]
    df['Sell_signal'] = x[1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.2], vertical_spacing=0.2,
        subplot_titles=['Price Chart', 'On Balance Volume(OBV)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name=f'ema_{ema}'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='grey', width=2), name='OBV'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['OBV_EMA'], line=dict(color='goldenrod', width=2), name='OBV_EMA'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
        marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
        marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='On Balance Volume(OBV) Trading Strategy', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

#Bollinger Bands MFI Strategy
def bollinger_mfi(bands, std, mfi):
    df = yf.download(stock, period=period)
    df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=mfi, fillna=True).money_flow_index()
    df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close'], window=bands, window_dev=std, fillna=True).bollinger_hband()
    df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close'], window=bands, window_dev=std, fillna=True).bollinger_lband()

    def strategy(df):
        buy_price = []
        sell_price = []
        flag = 0
        price = 0

        for i in range(len(df)):
            if df['Bollinger_Low'][i] > df['Close'][i] and df['MFI'][i] > 20:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.nan)
                    flag = 1
                    price = df['Close'][i]
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            elif df['Bollinger_High'][i] < df['Close'][i] and df['MFI'][i] > 60 and price < df['Close'][i]:
                if flag != -1 and flag != 0:
                    buy_price.append(np.nan)
                    sell_price.append(df['Close'][i])
                    flag = -1
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)

        return buy_price, sell_price

    buy, sell = strategy(df)

    df['Buy_signal'] = buy
    df['Sell_signal'] = sell

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.5,1.2], vertical_spacing=0.2, subplot_titles=['Price Chart', 'Money Flow Index(MFI)'])
    fig.add_trace(go.Candlestick(x=df.index,  open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='grey', width=2), name='MFI'), row=2, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Buy_signal'], marker_color='green', mode='markers+text',
        marker=dict(symbol='triangle-up', size=10), name='Buy'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Sell_signal'], marker_color='red', mode='markers+text',
        marker=dict(symbol='triangle-down', size=10), name='Sell'), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Bollinger_High'], line_color='gray', line={'dash': 'dash'}, name='upper band', opacity=0.5), row=1, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['Bollinger_Low'], line_color='gray', line={'dash': 'dash'}, fill='tonexty', name='lower band', opacity=0.5), row=1, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
    fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
    layout = go.Layout(font_family='Monospace', titlefont_size=20, xaxis_rangeslider_visible=True, title='Bollinger Bands MFI Strategy', height=1000)
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

strat = st.sidebar.multiselect("Trading Strategies", ['Bollinger Bands MFI Strategy','Double EMA Strategy', 'Double SMA Strategy',\
    'Exponential Moving Average(EMA) Strategy', 'Fibonacci Retracement Strategy', 'MACD Strategy', 'On Balance Volume(OBV) Strategy',\
    'RSI-EMA Strategy', 'Stochastics-EMA Strategy', 'Stochastics-MACD-RSI Strategy'])

if 'Bollinger Bands MFI Strategy' in strat:
    mfi = st.sidebar.slider("Money Flow Index", 5,100,25)
    bands = st.sidebar.slider("Bollinger Bands Perido", 5,100,20)
    std = st.sidebar.slider('Bollinger Band Standard Deviation',1,20,2)
    bollinger_mfi(bands, std, mfi)
if 'Fibonacci Retracement Strategy' in strat:
    df = fib_ret_strategy(df)
if 'Double SMA Strategy' in strat:
    sma1 = st.sidebar.slider('Simple Moving Average 1', 5, 250, 10)
    sma2 = st.sidebar.slider('Simple Moving Average 2', 10, 250, 15)
    dsma_strategy(df, sma1, sma2)
if 'Exponential Moving Average(EMA) Strategy' in strat:
    ema1 = st.sidebar.slider('Exponential Moving Average 1', 5, 250, 5)
    ema_strategy(df, ema1)
if 'Double EMA Strategy' in strat:
    dema1 = st.sidebar.slider('Double-Exponenting Moving Average 1', 5, 250, 10)
    dema2 = st.sidebar.slider('Double-Exponenting Moving Average 2', 10, 250, 30)
    dema_strategy(df, dema1, dema2)
if 'Stochastics-EMA Strategy' in strat:
    stoch = st.sidebar.slider('Stochastics',5, 50, 20)
    stochema = st.sidebar.slider('STOCH-EMA',5, 200, 5)
    swin = st.sidebar.number_input('K% Smooth', step=1, min_value=1)
    swin1 = st.sidebar.number_input('D% Smooth', step=1, min_value=1)
    stoch_ema(df, stoch, stochema, swin, swin1)
if 'RSI-EMA Strategy' in strat:
    rsi = st.sidebar.slider('Relative Strength Index(RSI)',2,35,14)
    rsiema = st.sidebar.slider('RSI-EMA',5,100,20)
    rsi_ema(df, rsi, rsiema)
if 'Stochastics-MACD-RSI Strategy' in strat:
    stoc = st.sidebar.slider('STOCHASTICS',5,50,5)
    stoch_macdf = st.sidebar.slider('MACD-FAST',5,50,12)
    stoch_macds = st.sidebar.slider('MACD-SLOW',5,50,26)
    sto_signal = st.sidebar.slider('MACD-SIGNAL',5,50,9)
    stoch_rsi = st.sidebar.slider('RSI',2,35,14)
    stoch_macd_rsi(stoc, stoch_macdf, stoch_macds, sto_signal, stoch_rsi)
if 'MACD Strategy' in strat:
    macd = st.sidebar.slider('MACD FAST',5,50,12)
    slow = st.sidebar.slider('MACD SLOW',5,50,26)
    signal = st.sidebar.slider('MACD SIGNAL',5,50,9)
    macd_strat(df, macd, slow, signal)
if 'On Balance Volume(OBV) Strategy' in strat:
    obv_ema = st.sidebar.slider('OBV_EMA',5,100,20)
    o_ema = st.sidebar.slider('EMA for OBV',5,200,10)
    obv_strategy(df, obv_ema, o_ema)

quote = yf.download(stock, period=period)

def open_pred(quote):
    model = load_model('model/open_pred.h5')
    #Get the quote
    quote = quote

    #Creating a new dataframe
    new_df = quote.filter(['Open'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_df)

    #Getting the last 60 days closing price Values and converting the data frame to an array
    last_60_days = new_df[-60:].values

    #Scale the data to be the values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    #Create an empty List
    X_test = []

    #Append the past 60 days
    X_test.append(last_60_days_scaled)

    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Get the predicted scaled price
    pred_price = model.predict(X_test)

    #Undoing the Scaling
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price

def high_pred(quote):
    model = load_model('model/high_pred.h5')
    #Get the quote
    quote = quote

    #Creating a new dataframe
    new_df = quote.filter(['High'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_df)

    #Getting the last 60 days closing price Values and converting the data frame to an array
    last_60_days = new_df[-60:].values

    #Scale the data to be the values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    #Create an empty List
    X_test = []

    #Append the past 60 days
    X_test.append(last_60_days_scaled)

    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Get the predicted scaled price
    pred_price = model.predict(X_test)

    #Undoing the Scaling
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price

def low_pred(quote):
    model = load_model('model/low_pred.h5')
    #Get the quote
    quote = quote

    #Creating a new dataframe
    new_df = quote.filter(['Low'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_df)

    #Getting the last 60 days closing price Values and converting the data frame to an array
    last_60_days = new_df[-60:].values

    #Scale the data to be the values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    #Create an empty List
    X_test = []

    #Append the past 60 days
    X_test.append(last_60_days_scaled)

    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Get the predicted scaled price
    pred_price = model.predict(X_test)

    #Undoing the Scaling
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price

def close_pred(quote):
    model = load_model('model/close_pred.h5')
    #Get the quote
    quote = quote

    #Creating a new dataframe
    new_df = quote.filter(['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_df)

    #Getting the last 60 days closing price Values and converting the data frame to an array
    last_60_days = new_df[-60:].values

    #Scale the data to be the values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    #Create an empty List
    X_test = []

    #Append the past 60 days
    X_test.append(last_60_days_scaled)

    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Get the predicted scaled price
    pred_price = model.predict(X_test)

    #Undoing the Scaling
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price

pred_price = st.sidebar.selectbox('Select the type of Price Prediction', ['Close Price', 'OHLC(Open, High, Low, Close) Price'])
if pred_price == 'Close Price':
    if st.sidebar.button(f'Predict Closing Price for {StockName}'):
        df = yf.download(stock, period=period)
        pred = close_pred(quote)
        price = int(pred)

        day = datetime.now().strftime('%A')
        now = datetime.now().replace(microsecond=0)
        mopen = now.replace(hour=9, minute=15, second=0, microsecond=0)
        mclose = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if day == "Monday" or day == "Tuesday" or day == 'Wedensday' or day == 'Thursday' or day == 'Friday':
            if now >= mopen and now <= mclose:
                st.info('Market Open')
                prev = df.iloc[-2]['Close']
            else:
                st.info('Market Closed')
                prev = df.iloc[-1]['Close']
        else:
            prev =  df.iloc[-1]['Close']

        diff = round(price - prev, 2)
        delta = diff
        label = f'Close Price for {StockName}'
        st.metric(label=label, value=price, delta=delta)
        if diff > 0:
            st.subheader('The price will go Up by Rs {}'.format(str(diff)))
        else:
            diff = round(prev - price, 2)
            st.subheader('The price will go Down by Rs.{}'.format(str(diff)))
        st.write('Closing Price for Previous Trading Day Was Rs {}'.format(round(prev,2)))
        st.info('The prices predicted may vary from Rs.20 to Rs.40')

if pred_price == 'OHLC(Open, High, Low, Close) Price':
    if st.sidebar.button(f'Predict OHLC Price for {StockName}'):
        df = yf.download(stock, period=period)
        open = open_pred(quote)
        high = high_pred(quote)
        low = low_pred(quote)
        close = close_pred(quote)
        open, high, low, close = int(open), int(high), int(low), int(close)

        day = datetime.now().strftime('%A')
        now = datetime.now().replace(microsecond=0)
        mopen = now.replace(hour=9, minute=15, second=0, microsecond=0)
        mclose = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if day == "Monday" or day == "Tuesday" or day == 'Wednesday' or day == 'Thursday' or day == 'Friday':
            if now >= mopen and now <= mclose:
                st.info('Market Open')
                oprice, cprice = df.iloc[-2]['Open'], df.iloc[-2]['Close']
            else:
                st.info('Market Closed')
                oprice, hprice, lprice, cprice = df.iloc[-1]['Open'], df.iloc[-1]['High'], df.iloc[-1]['Low'], df.iloc[-1]['Close']
        else:
            st.warning('Error!')

        odiff = round(open - cprice, 2)
        hdiff = round(high - open, 2)
        ldiff = round(low - open, 2)
        cdiff = round(close - open, 2)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=f'Open Price for {StockName}', value=open, delta=odiff)
            st.write('Closing Price for Previous Trading Day Was Rs.{}'.format(round(cprice,2)))
            if odiff < 0:
                st.write('There will be Downside Opening')
            else:
                st.write('There will be a Upside Opening')
        with col2:
            st.metric(label=f'High Price for {StockName}', value=high, delta=hdiff)
            st.write('High Price for Current Trading Session can be upto {} Points from the Stock Open Price'.format(round(hdiff,2)))
        with col3:
            st.metric(label=f'Low Price for {StockName}', value=low, delta=ldiff)
            if ldiff < 0:
                ldiff = round(open - low, 2)
            st.write('Low Price for Current Trading Session can be upto {} Points from the Stock Open Price'.format(round(ldiff,2)))
        with col4:
            st.metric(label=f'Close Price for {StockName}', value=close, delta=cdiff)
            if cdiff < 0:
                cdiff = round(open - close, 2)
                st.write('Closing Price for Current Trading Day can be {} points Down from Stock Open Price'.format(round(cdiff,2)))
                st.write("The Closing will de Downside")
            else:
                st.write('Closing Price for Current Trading Day can be {} points Up from Stock Open Price'.format(round(cdiff,2)))
                st.write('The Closing will be Upside')
        st.info('Note : The prices predicted may vary from 10 to 30 Points in the Actual Stock Prices')