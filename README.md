# House_Price_Prediction_using_ML
In these model, I use data set of "kc_House_data" and perform Linear Regression which give me the output of 73%



#  House Price Predicition using Machine Learning (ML)

##  Linear Regression model on kc_House_Data 

Firstly, we import our libraries and dataset and then we see the head of the data to know how the data looks like and use describe function to see the percentile’s and other key statistics.

import numpy as np       ###linear algebra
import pandas as pd     ###datapre-processing
from matplotlib import pyplot as plt    
%matplotlib inline
import seaborn as sns

df = pd.read_csv(r"C:\Users\Sikkim\Downloads\kc_house_data.csv")
df.info()

df.head()

df.describe()

Now , we are going to see some visualization and also going to see how and what can we infer from visualization.

df['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number Of bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')


df.corr()

As we can see from the visualization 3 bedroom houses are most commonly sold followed by 4 bedroom.

sns.heatmap(df.corr())

### Visualizing the location of the houses based on latitude and longitude.

So according to the dataset , we have latitude and longitude on the dataset for each house. We are going to see the common location and how the houses are placed.
We use seaborn , and we get his beautiful visualization. Joinplot function helps us see the concentration of data and placement of data and can be really useful.

plt.figure(figsize=(10,10))
sns.jointplot(x=df.lat.values, y=df.long.values, size=10)
plt.xlabel('Latitude', fontsize=12)
plt.ylabel('Longitude', fontsize = 12)
plt.show()
##sns.despine

For latitude between -47.7 and -48.8 there are many houses , which would mean that maybe it’s an ideal location isn’t it ? But when we talk about longitude we can see that concentration is high between -122.2 to -122.4.

#### How common factors are affecting the price of the houses ?

catter plot helps us to see how our data points are scattered and are usually used for two variables. From the first figure we can see that more the living area , more the price though data is concentrated towards a particular price zone , but from the figure we can see that the data points seem to be in linear direction.

plt.scatter(df.price,df.sqft_living)
plt.title("Price  Vs  Square feet")

 The second figure tells us about the location of the houses in terms of longitude and it gives us quite an interesting observation that -122.2 to -122.4 sells houses at much higher amount.

plt.scatter(df.price, df.long)
plt.title('Price  Vs location of the Area')

We can see more factors affecting the price


plt.scatter(df.bedrooms, df.price)
plt.title('Bedroom Vs Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()

plt.scatter((df['sqft_living']+df['sqft_basement']),df['price'])

plt.scatter(df.waterfront, df.price)
plt.title('Waterfront  Vs  price')

plt.scatter(df.floors, df.price)

plt.scatter(df.condition, df.price)

plt.scatter(df.zipcode, df.price)

### Linear Regression 

In easy words a model in statistics which helps us predicts the future based upon past relationship of variables. 

Regression works on the line equation , y=mx+c , trend line is set through the data points to predict the outcome.

The variable we are predicting is called the criterion variable and is referred to as Y. The variable we are basing our predictions on is called the predictor variable and is referred to as X. When there is only one predictor variable, the prediction method is called Simple Regression.

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

labels = df['price']
conv_dates = [1 if values == 2014 else 0 for values in df.date]
df['date'] = conv_dates
train1 = df.drop(['id', 'price'], axis = 1)

from sklearn.model_selection import train_test_split

We use train data and test data , train data to train our machine and test data to see if it has learnt the data well or not.

Now we know that prices are to be predicted , hence we set labels (output) as price columns and we also convert dates to 1’s and 0’s so that it doesn’t influence our data much . We use 0 for houses which are new that is built after 2014.

We have made my train data as 90% and 10% of the data to be my test data , and randomized the splitting of data by using random_state.

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size = 0.10, random_state = 2)
reg.fit(x_train, y_train)

reg.score(x_test, y_test)

After fitting our data to the model we can check the score of our data i.e., in this case the prediction is 73%
