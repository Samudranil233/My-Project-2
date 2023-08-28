#LIBRARIES THAT ARE NEED TO BE IMPORTED 
# In[3]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ### Data Sets ###

# In[6]:


wine = "winequality-red.csv"


# In[7]:


data = pd.read_csv(wine)


# In[50]:


##Given Data
data


# In[52]:


#shape of data
data.shape


# In[12]:


#data information
data.info()


# In[13]:


#checking whether data has any null values or not.
data.isnull().sum()


# In[54]:


#checking data distribution
data.head(8)


# In[55]:


data.tail()


# In[56]:


#the Data Analysis

data.describe()


# ### Graphs for Insights ###

# In[57]:


sns.catplot(x="quality", data = data, kind="count")


# In[58]:


#taking x and y as chlorides versus pH

plot = plt.figure(figsize =(5,5))
sns.barplot(x='chlorides', y="pH", data = data)
plt.show()


# In[59]:


#same goes on with quality vs citric acid

plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y="citric acid", data = data)
plt.show()


# In[60]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='free sulfur dioxide', y="free sulfur dioxide", data = data)
plt.show()


# In[61]:


correlation = data.corr()


# In[62]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar = True, square=True, fmt=".1f", annot = True, annot_kws={"size":8}, cmap ='Blues')


# # Data Pre Processing 

# In[63]:


x = data.drop("quality", axis = 1)


# In[64]:


x


# In[81]:


y = data["quality"].apply(lambda y_value:1 if y_value>=7 else 0)


# In[66]:


y


# In[67]:


#splitting
#Data Pre processing

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=3)


# In[68]:


x_train.shape


# In[69]:


x_test.shape


# In[70]:


y_train.shape


# In[79]:


y_test.shape


# # Random Forest Classifier Model

# In[71]:


##model training


model = RandomForestClassifier()


# In[72]:


#model 
model.fit(x_train, y_train)


# In[73]:


x_test_predictions = model.predict (x_test)


# In[74]:


test_data_accuracy = accuracy_score(x_test_predictions, y_test)


# In[75]:


test_data_accuracy = accuracy_score(x_test_predictions, y_test)


# In[76]:


print('data accuracy:', test_data_accuracy)


# In[80]:


#making the system capable of making predictions
#a predictive system

input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
array = np.asarray(input_data)
reshape = array.reshape(1,-1)
predictions = model.predict(reshape)
print (predictions)

if predictions[0]==1:
    print("Wine is of Good Quality")
else:
    print('Wine is of Bad Quality')


# In[ ]:





# # Thank You 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




