#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report


# In[5]:


df_true = pd.read_csv("fakeNews/True.csv")
df_fake = pd.read_csv("fakeNews/Fake.csv")


# In[6]:


df_true['class'] = 1
df_fake['class'] = 0


# In[11]:


df = pd.concat([df_true,df_fake],axis = 0)


# In[12]:


df


# In[13]:


df1 = df[['text','class']]


# In[17]:


import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[18]:


df1['text'] = df1['text'].apply(wordopt)


# In[ ]:





# In[19]:


X = df1['text']
Y = df1['class']


# In[20]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 42,random_state = 42)


# In[22]:


tf_idf = TfidfVectorizer(stop_words ='english',max_df = 0.7)
x_train = tf_idf.fit_transform(x_train)
x_test = tf_idf.transform(x_test)


# In[26]:


regressor = LogisticRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


# In[28]:


print("accuracy_score: ",accuracy_score(y_pred,y_test))


# In[29]:


print("classification report: ",classification_report(y_pred,y_test))


# In[ ]:




