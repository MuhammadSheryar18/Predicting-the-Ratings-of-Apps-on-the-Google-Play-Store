#!/usr/bin/env python
# coding: utf-8

# ### Google Play Store DataSet.

# #### Problem Statement:
# "Predicting the Ratings of Apps on the Google Play Store Based on Their Characteristics and User Reviews."

# In[17]:


import re
import sys

import time
import scipy
import datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[18]:


df = pd.read_csv(r'C:\Users\ibrahim laptops\Downloads\googleplaystore.csv')


# The dataset, "googleplaystore.csv," contains detailed information about various apps on the Google Play Store. The columns include:
# 
# #### App: 
# The name of the application.
# #### Category: 
# The category under which the app is listed.
# #### Rating: 
# The overall user rating of the app.
# #### Reviews: 
# The number of user reviews.
# #### Size: 
# The size of the app.
# #### Installs: 
# The number of times the app has been installed.
# #### Type: 
# Whether the app is free or paid.
# #### Price: 
# The price of the app (if it's a paid app).
# #### Content Rating: 
# The intended audience or age group for the app.
# #### Genres: 
# The genre of the app.
# #### Last Updated: 
# The date when the app was last updated.
# #### Current Ver: 
# The current version of the app.
# #### Android Ver: 
# The minimum Android version required to run the app.

# ### Data Inspection

# In[19]:


df.head()


# In[20]:


df.describe()


# In[21]:


df.info()


# In[22]:


df.dtypes


# In[23]:


df.isnull().sum()


# In[24]:


# Exploring missing data and checking if any has NaN values
sns.set_palette("pastel")
plt.figure(figsize=(7, 5))
sns.heatmap(df.isnull())


# #### Number of missing values in each column
# 
# Rating: 1,474 missing values.
# 
# Type: 1 missing value.
# 
# Content Rating: 1 missing value.
# 
# Current Ver: 8 missing values.
# 
# Android Ver: 3 missing values.
# 
# There are missing values in "Rating", "Type", "Content Rating", "Current Ver" and " Android Ver". But most of these missing values in Rating column.

# In[25]:


df.duplicated().sum()


# In[26]:


df.shape


# ### Data Cleaning

# In[27]:


# Converting 'Reviews' to numeric (integers)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')


# In[28]:


#creating a function to change these values and convert them into float

def convert_size(size):
    if isinstance (size, str):
        if 'Varies with device' in size:
            return 0
        elif 'M' in size:
            return float(size.replace('M', ''))*(1024*1024)
        elif 'k' in size:
            return float(size.replace('k',''))*1024
    return size
#applying the funciton

df['Size']= df['Size'].apply(convert_size)
# renaming the column as we converted the size into bytes
df.rename(columns={'Size': 'Size_in_bytes'}, inplace=True)
df['Size_in_bytes']


# In[30]:


# Converting 'Installs' to numeric: removing '+' and ',' then converting to integers
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')


# In[31]:


# Converting 'Price': removing '$' if present and converting to float
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')


# In[33]:


df.dtypes


# We merge app details with user reviews, convert numeric-like columns ('Reviews', 'Size', 'Installs', 'Price') from string to numeric formats, and standardize the 'Size' column into megabytes for consistency. These steps prepare the dataset for predictive modeling of app ratings.

# In[34]:


# The best way to fill missing values might be using the median instead of mean.
df['Rating'] = df['Rating'].fillna(df['Rating'].median())


# In[35]:


# Before filling null values we have to clean all non numerical values & unicode charachters
replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
for i in replaces:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
for j in regex:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)
df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())


# Two strategies to handle missing data, either removing records with these missing values or replacing missing values with a specific value like (mean, median or mode) value of the column.

# In[36]:


# Count the number of unique values in category column
df['Category'].unique()


# In[37]:


# Removing NaN values
df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]


# In[39]:


# Convert 'Installs' to numeric, setting errors to NaN
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Specify the bin edges
bin_edges = [-1, 0, 100, 10000, 100000, 1000000, 10000000, 1000000000]

# Define bin labels
bin_labels = ['no', 'Very Low', 'Low', 'Average', 'More than Average', 'High', 'Very High']

# Now apply pd.cut()
df['Install_category'] = pd.cut(df['Installs'], bins=bin_edges, labels=bin_labels)

# Continue with your analysis
df['Install_category'].value_counts()


# In[40]:


df.isnull().sum().sort_values(ascending=False)


# In[42]:


# remove the rows having null values in the 'Android Ver', 'Type'
df.dropna(subset=['Android Ver',  'Type'], inplace=True)


# In[43]:


df.isnull().sum()


# ### Data Analysis

# In[45]:


sns.pairplot(df,hue='Installs')


# In[47]:


sns.barplot(x=df.Type.value_counts().index,y=df.Type.value_counts())


# In[48]:


sns.barplot(x=df['Content Rating'].value_counts().index, y=df['Content Rating'].value_counts())


# In[51]:


plt.figure(figsize=(10,5))
sns.boxplot(x="Installs", y="Rating", data=df)


# In[59]:


# relationships between 'Rating' and other key variables
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
sns.histplot(df['Rating'], bins=20, kde=True, ax=axes[0, 0])
sns.histplot(df['Size_in_bytes'], bins=20, kde=True, ax=axes[0, 0])
sns.histplot(np.log1p(df['Reviews']), bins=20, kde=True, ax=axes[0, 1])
sns.histplot(np.log1p(df['Installs']), bins=20, kde=True, ax=axes[1, 0])
sns.histplot(df['Price'], bins=20, kde=True, ax=axes[1, 1])


# In[63]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
sns.scatterplot(x='Reviews', y='Rating', data=df, ax=axes[0, 0], alpha=0.5)
sns.scatterplot(x='Size_in_bytes', y='Rating', data=df, ax=axes[0, 1], alpha=0.5)
sns.scatterplot(x='Installs', y='Rating', data=df, ax=axes[1, 0], alpha=0.5)
sns.scatterplot(x='Price', y='Rating', data=df, ax=axes[1, 1], alpha=0.5)


# In[61]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=2)


# ### Categorical Data Encoding¶
# Many machine learning algorithms can support categorical values without further manipulation but there are many more algorithms that do not. We need to make all data ready for the model, so we will convert categorical variables (variables that stored as text values) into numircal variables.

# In[64]:


# App values encoding
le = preprocessing.LabelEncoder()
df['App'] = le.fit_transform(df['App'])
# This encoder converts the values into numeric values


# In[65]:


# Category features encoding
category_list = df['Category'].unique().tolist()
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)


# In[66]:


# Genres features encoding
le = preprocessing.LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])


# In[67]:


# Encode Content Rating features
le = preprocessing.LabelEncoder()
df['Content Rating'] = le.fit_transform(df['Content Rating'])


# In[68]:


# Last Updated encoding
df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))


# ### Evaluation Procedure

# In this section shows how k-nearest neighbors and random forests can be used to predict app ratings based on the other matrices. First, the dataset has to separate into dependent and independent variables (or features and labels). Then those variables have to split into a training and test set.
# 
# During training stage we give the model both the features and the labels so it can learn to classify points based on the features.

# In[70]:


# Split data into training and testing sets
features = ['App', 'Reviews', 'Size_in_bytes', 'Installs', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
features.extend(category_list)
X = df[features]
y = df['Rating']


# In[71]:


X.head()


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)


# The above script splits the dataset into 75% train data and 25% test data.

# ### Linear regression
# Linear regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables. The goal is to find a linear equation that best predicts the dependent variable based on the values of the independent variables.

# In[73]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[74]:


# Creating a Linear Regression model
model = LinearRegression()

# Fitting the model with the training data
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[75]:


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# Mean Squared Error (MSE) = 0.2355414883378792:
# 
# MSE is a measure of the average squared difference between the observed actual outturns and the values predicted by the model. A lower MSE value is generally better, as it indicates that the model's predictions are closer to the actual data. An MSE of approximately 0.23 suggests that, on average, the squared difference between the predicted and actual values is around 0.23.
# 
# Mean Absolute Error (MAE) = 0.3229683136358806:
# 
# MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight. A MAE of approximately 0.32 means that, on average, the absolute error of your predictions is around 0.32 units from the actual values.
# 
# R-squared = -0.0006600595998533798:
# 
# R-squared is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in a regression model. R-squared values range from 0 to 1, where a higher value generally indicates a better fit of the model to the data. A value close to 1 suggests that a large proportion of the variance in the dependent variable has been explained by the regression model.

# ### K-Nearest Neighbors Model
# The k-nearest neighbors algorithm is based around the simple idea of predicting unknown values by matching them with the most similar known values. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset — its "nearest neighbors".

# In[76]:


# Look at the 15 closest neighbors
model = KNeighborsRegressor(n_neighbors=15)


# In[77]:


# Find the mean accuracy of knn regression using X_test and y_test
model.fit(X_train, y_train)


# In[78]:


# Calculate the mean accuracy of the KNN model
accuracy = model.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'


# In[79]:


# Try different numbers of n_estimators - this will take a minute or so
n_neighbors = np.arange(1, 20, 1)
scores = []
for n in n_neighbors:
    model.set_params(n_neighbors=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neighbors, scores)


# ### Random Forest Model
# The RandomForestRegressor class of the sklearn.ensemble library is used to solve regression problems via random forest. The most important parameter of the RandomForestRegressor class is the n_estimators parameter. This parameter defines the number of trees in the random forest.

# In[80]:


model = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results


# In[81]:


predictions = model.predict(X_test)
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)


# In[83]:


'Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)


# In[84]:


'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




