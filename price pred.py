import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
yf.pdr_override()


share = input('Enter Name of the Stock : ')
#load the trained model
model = load_model('model/stockPrediction.h5')

#Get the quote
quote = yf.download(share, period='max')

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
print(pred_price)