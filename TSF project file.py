#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION GRIP #Task1

# # Prediction using Supervised ML (Level - Beginner)

#          BY RUPESH KUMAR (DS & BI Intern)

# # Problem statement

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# # To predict:

# What will be predicted score if a student studies for 9.25 hrs/ day?

# # Problem Soution

# # Step:1 --- Importing all libraries required in this notebook

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# # Step:2--- Reading data from csv file and visualization¶

# In[5]:


# Reading the data
path = r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(path)


# In[6]:


data.head()


# In[7]:


# Check if there any null value in the Dataset
data.isnull == True


# # There is no null value in the Dataset so, we can now visualize our Data.

# In[ ]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# # From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied', Lets plot a regression line to confirm the correlation.

# In[8]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# # It is confirmed that the variables are positively correlated.

# # Training the Model

# 1) Splitting the Data

# In[9]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# 2) Fitting the Data into the model

# In[10]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# # Predicting the Percentage of Marks

# In[11]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# # Visually Comparing the Predicted Marks with the Actual Marks
# 

# In[12]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# # Evaluating the Model

# In[13]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# Small value of Mean absolute error states that the chances of error or wrong forecasting through the model are very less.

# #  What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[14]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.

# In[ ]:




