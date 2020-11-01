#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

df = pd.read_csv('car.csv')


# In[2]:


df.head()


# In[9]:


features=df.iloc[:,0:6]
features.head()


# In[12]:


labels=df.iloc[:,6:]
labels.head()


# In[14]:


#The features are the descriptive attributes, and the label is what you're attempting to predict or forecast.
#features=("buying","maint","doors","persons","lug_boot","safety")
#labels=("Class Values")
#make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices((features,labels))


# In[15]:


# using a tensor
dataset = tf.data.Dataset.from_tensor_slices(df)

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))


# In[ ]:




