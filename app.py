import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2009-12-31'
end = '2024-03-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
data = yf.download(user_input, start, end)

#Describing data
st.subheader('Data from 2009 to 2024')
st.write(data.describe())

#visualisation

st.subheader('Closing Price v/s Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Timw Chart with 100 MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price v/s Timw Chart with 100 MA & 200 MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)


#splitting the data into training and testing
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])

print(data_test.shape)
print(data_train.shape)

#converting data to 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)



#Loading my model:

model = load_model('keras_model.h5')

# Testing part
past_hundred_days = data_train.tail(100)
final_df = pd.concat([past_hundred_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scalar = scaler.scale_

scale_factor = 1/scalar[0]
y_predicted *= scale_factor
y_test *= scale_factor

#Graphical Visualisation
st.subheader('Predicted vs Original prices')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


