import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import streamlit as st
from keras.models import load_model
from keras.optimizers import Adam
start = '2010-01-01'
end = '2023-10-20'

st.title('Stock Price Analysis')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

yfin.pdr_override()
df = pdr.get_data_yahoo(user_input, start, end)

# Describing data
st.subheader('Data from 2010 to 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()  # Add () to call the method
fig = plt.figure(figsize=(12, 6))

plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 Moving Average')
ma200 = df.Close.rolling(200).mean()  # Add () to call the method
fig = plt.figure(figsize=(12, 6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)




data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)


# #splitting data into xtrain and ytrain
# x_training=[]
# y_training=[]
# for i in range(100,data_train_array.shape[0]):
#     x_training.append(data_train_array[i-100:i])
#     y_training.append(data_train_array[i,0])
    
# x_training,y_training=np.array(x_training),np.array(y_training)


#load my model


custom_optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate as needed

# Register the custom optimizer in a dictionary
custom_objects = {'CustomAdam': custom_optimizer}

# Load your Keras model with the custom objects registered
model = load_model('keras_model.h5', custom_objects=custom_objects)

# The rest of your code
past_100 = data_train.tail(100)
final_df = pd.concat([past_100, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test_array, y_test_array = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test_array)

scaler = scaler.scale_
# Now, inverse the scaling
scale_factor = 1 / scaler[0]

y_predicted = y_predicted * scale_factor
y_test_array = y_test_array * scale_factor

# Final graph
st.subheader('Predictions vs Original Chart')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test_array, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
