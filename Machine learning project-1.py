#!/usr/bin/env python
# coding: utf-8

# # Price prediction
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# ## Train test splitting

# In[8]:


#TRAIN TEST SPLIT FUNCTION
#This function is self defined. it is already present in scikitlearn libary



import numpy as np
np.random.seed(42)

def split_train_test(data,test_ratio):
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    


# In[9]:


train_set, test_set = split_train_test(housing,0.2)


# In[10]:


#print(f"ROWS in the train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:



#TRAIN TEST SPLIT FUNCTION


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[12]:


print(f"ROWS in the train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['CHAS']):
    
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


376/28


# In[17]:


95/7


# In[18]:


housing = strat_train_set.copy()


# ## Looking For Correlations

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attribute = ["MEDV","RM", "ZN","LSTAT"]
scatter_matrix(housing[attribute], figsize=(12,8))


# In[21]:


housing.plot(kind= 'scatter', x='RM',y='MEDV',alpha=0.8)


# ## Trying out Attribute combinations

# In[22]:


housing["TAXRM"] = housing["TAX"]/housing["RM"]


# In[23]:


housing["TAXRM"]


# In[24]:


housing.head()


# In[25]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[26]:


housing.plot(kind= 'scatter', x='TAXRM',y='MEDV',alpha=0.8)


# In[27]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing ATTRIBUTE

# In[28]:


# To take care of the missing attributes, you have three options :

# 1. Get rid of the missing attribute
# 2. Get rid of the Whole attribute
# 3. Set the value to some value (0,mean and median)


# In[29]:


a = housing.dropna(subset=['RM']) #option 1
a.shape
# Note that the orginal housing dataframe will remain unchanged


# In[30]:


housing.drop("RM", axis=1).shape #option 2

# Note that there is no RM column and also note that the orginal housing dataframe will remain unchanged


# In[31]:


median = housing["RM"].median()
median


# In[32]:


housing["RM"].fillna(median) #Compute median for option 3

# Note that the orginal housing dataframe will remain unchanged


# In[33]:


housing.shape


# In[34]:


housing.describe() # before we started filling missing attributes


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer (strategy= "median")
imputer.fit(housing)


# In[36]:


imputer.statistics_


# In[37]:


X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns= housing.columns) # tr = Transformed dataset


# In[38]:


housing_tr.describe()


# housing_tr = pd.DataFrame(X, columns= housing.columns) # tr = Transformed dataset

# ## Scikit learn design

# Primarily,  three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. example- imputer. It has a fit method and transformer method.
# fit method - fits the dataset and calculates internal parameters 
# 
# 2. Transformers - transform method takes input and return output based on the learnings from fits(). It also has a convenience function called fit_transforms() which fits and then tranforms.
# 
# 
# 3. Predictors - LinearRegression model is example of the predictor. fit() and predict() are two common functions. It also gives score() function which will evalutes the predictions.

# ## feature scaling 

# Primarily, two types of feature scaling methods 
# 1. Min-max scaling (Normalization)
#    (value - min)/(max - min)
#    Sklearn provides a class called MinMaxScalar for this
#    
#    
# 2. Standardization 
#    (value - mean)/std
#    Sklearn provides a class called standard Scaler for this
#    

# ## Creating pipeline

# In[39]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    #add as many as you want in your pipeline
    ("std_scaler",StandardScaler()),
])


# In[40]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[41]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon real estates

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
#model = DecisionTreeRegressor()
#model = LinearRegression()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[44]:


some_data = housing.iloc[:5]


# In[45]:


some_labels = housing_labels.iloc[:5]


# In[46]:


prepared_data = my_pipeline.transform(some_data)


# In[47]:


model.predict(prepared_data)


# In[48]:


list(some_labels)


# ## Evaluating the model

# In[49]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[50]:


rmse


# ## using better evaluation techinque - cross validation 

# In[51]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[52]:


rmse_scores


# In[53]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("standard deviation:", scores.std())
    


# In[54]:


print_scores(rmse_scores)


# Quiz : Convert this notebook into python file run the pipeline using PYcharm

# ## Saving the model

# In[56]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# ## Testing the model on test data 

# In[63]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)

final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[61]:


final_rmse


# In[ ]:




