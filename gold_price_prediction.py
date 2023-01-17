#Problem statement for the project - 
#We need to build a machine learning algorithm to predict the price of gold based on the given parameters in the dataset.

#workflow 
#1. load gold data 
#2. data preprocessing 
#3. data exploration and analysis 
#4. train test split 
#5. model used - random forest regressor 

#load libraries
#linear algebra - building matrices  
import numpy as np 

#data preprocessing 
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

#data analysis and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 

#load data 
gold_data = pd.read_csv(r'gold_data.csv')

#view data 

#view first five rows of the dataset 
gold_data.head()
#the dataset contains 6 columns as follows - 
#1. Date 
#2. SPX
#3. GLD - is the largest ETF to invest directly in physical gold
#4. USO
#5. SLV
#6. EUR/USD

#view the total number of rows and columns 
gold_data.shape
#2290 rows (2290 data points) and 6 columns 

#view statistical measures 
gold_data.describe()

#get more insight into the type of columns 
gold_data.info()

#view last 5 rows 
gold_data.tail()

#view missing values 
gold_data.isnull().sum()
#the dataset doesn't have any missing values 

#always check correlation in case of regression 
#correlation 
correlation = gold_data.corr()
#contruct heat map 
plt.figure(figsize=(8,8))
sns.heatmap(correlation, char=True, fmt='.if', annot=True, annot_kws={'size':8}, cmap='Blues')

#correlation values of gld
print(correlation['GLD'])

#check distribution at gld price 
sns.displot(gold_data['GLD'],color='green')

#separate data and label, feature and target 
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)

#train model 
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)

#evaluate model 
test_data_prediction = regressor.predict(X_test)
print (test_data_prediction)

#r squared error 
error_score = metrics.r2_score(Y_test, test_data_prediction)
print ('r squared error', error_score)
#the r squared error is 0.98

#compare actual and predicted values 
Y_test = list(Y_test)
plt.plot(Y_test, color='blue',label='Actual Value')
plt.plot(test_data_prediction, color='green', label='predicted value')
plt.title('actual price vs predicted price')
plt.xlabel('number of values')
plt.ylabel('GLD price')
plt.legend()
plt.show()
