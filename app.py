import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock Price Prediction')
st.text('Made with ❤️ by Pradhumna Guruprasad')

stocksTicker = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
stocksTicker = stocksTicker.reset_index()
stocksTicker = stocksTicker.drop(
    ['Exchange', 'Category Name', 'Country'], axis=1)
ticker = []

for i in range(stocksTicker.shape[0]):
    ticker.append(str(stocksTicker['Ticker'].values[i]) +
                  '-' + str(stocksTicker['Name'].values[i]))


option = st.sidebar.selectbox(
    'Select one symbol', tuple(ticker))
option = option.split('-')[0]
today = datetime.date.today()
before = today - datetime.timedelta(days=7000)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' %
                       (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

#userInput = st.text_input('Enter a Stock Ticker: ', 'AAPL')

try:

    df = data.DataReader(option, 'yahoo', start_date, end_date)

    st.subheader(f'{option} Stock Price from {start_date} to {end_date}')
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
    plt.legend(['100 Day Moving Average',
               '200 Day Moving Average', 'Closing Price'])
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

except:
    st.error('Error: Please enter a valid ticker symbol.')
