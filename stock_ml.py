# machine learning classification
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score


# for data manipulation
import pandas as pd
import numpy as np


# for plotting graphs
import matplotlib.pyplot as plt
import seaborn


# to fetch the data from online source
from pandas_datareader import data as pdr



# downloading the data from yahoo finance, using pandas_datareader
df=pdr.get_data_yahoo('SPY', start="2012-01-01", end= "2017-10-01")


#dropping empty values
df=df.dropna()


#plotting the graph on the basis of close price series of the stock
df.Close.plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()



# logic to determine the target variable is, whether S&P500 price will close up or close down on the next trading day,
# if the  if next trading day’s close price is greater than today’s close price then,
# we will buy the S&P500 index, else we will sell the S&P500 index. We will store +1 for the buy signal and -1 for the sell signal.
y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)


# X is a dataset that holds the predictor’s variables which are used to predict
# target variable, ‘y’. The X consists of variables such as ‘Open – Close’ and
# ‘High – Low’. These can be understood as indicators based on which the algorithm
# will predict the option price.
df['Open-Close']= df.Open-df.Close
df['High-Low']=df.High-df.Low
x=df[['Open-Close','High-Low']]



# first 80% data is used for training and remaining data for testing
# x_train and y_train are train dataset
# x_test and y_test are test dataset
split_percentage=0.8
split=int(split_percentage*len(df))


# train dataset
x_train=x[:split]
y_train=y[:split]




# test datset
x_test=x[split:]
y_test=y[split:]




# machine learning classification model using train dataset
cls= SVC().fit(x_train, y_train)



# computing the accuracy of the classification model on the train and test dataset,
# by comparing the actual values of the trading signal with the predicted values of the trading signal.
# The function accuracy_score() will be used to calculate the accuracy.
accuracy_train= accuracy_score(y_train, cls.predict(x_train))
accuracy_test= accuracy_score(y_test, cls.predict(x_test))



# Predicting the signal (buy or sell) for the test data set, using the cls.predict() function.
df['Predicted_signal']=cls.predict(x)


#strategy returns based on the signal predicted by the model in the test dataset.
# We save it in the column ‘Strategy_Return’ and then, plot the cumulative strategy returns.
df['Return']= np.log(df.Close.shift(-1)/df.Close)*100
df['Strategy_return']=df.Return*df.Predicted_signal
df.Strategy_return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns %")
plt.show()

