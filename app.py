from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Prediction')
st.text('Made with ❤️ by Pradhumna Guruprasad')

userInput = st.text_input('Enter a Stock Ticker: ', 'AAPL')
df = data.DataReader(userInput, 'yahoo', start, end)

st.subheader(f'{userInput} Stock Price from {start} to {end}')
st.write(df.describe())

# Visualize the data

st.subheader('Closing Price vs Time Chart')
ma100 = df.Close.rolling(window=100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(['100 Day Moving Average', 'Closing Price'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart')
ma100 = df.Close.rolling(window=100).mean()
ma200 = df.Close.rolling(window=200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(['100 Day Moving Average', '200 Day Moving Average', 'Closing Price'])
st.pyplot(fig)


# Splitting the data into training and testing sets

dataTrain = pd.DataFrame(df['Close'][0:int(df.shape[0]*0.7)])
dataTest = pd.DataFrame(df['Close'][int(df.shape[0]*0.7):int(df.shape[0])])

scaler = MinMaxScaler(feature_range=(0, 1))
dataTrainArray = scaler.fit_transform(dataTrain)

xTrain = []
yTrain = []

for i in range(100, dataTrainArray.shape[0]):
    xTrain.append(dataTrainArray[i-100:i])
    yTrain.append(dataTrainArray[i, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)

# Loading the model

model = load_model('keras_model.h5')

past_100 = dataTrain.tail(100)
finalDF = past_100.append(dataTest, ignore_index=True)
inputData = scaler.fit_transform(finalDF)

xTest = []
yTest = []

for i in range(100, inputData.shape[0]):
    xTest.append(inputData[i-100:i])
    yTest.append(inputData[i, 0])

xTest, yTest = np.array(xTest), np.array(yTest)

yPred = model.predict(xTest)

scaleFactor = 1/scaler.scale_[0]
yPred = yPred*scaleFactor
yTest = yTest*scaleFactor

st.subheader('Prediction vs Actual')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(yPred, 'r', label='Prediction')
plt.plot(yTest, 'b', label='Original')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)
