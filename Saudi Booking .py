#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as ex
import seaborn as sns 


# In[2]:


df=pd.read_csv('/home/faris/StackOverFlow/booking_saudi_cities_cleaned.csv')


# In[3]:


#check for dublicate rows
sum(df.duplicated())


# In[4]:


df.info()


# In[5]:


#Exloration data
print(df.shape[0])
print(df.shape[1])


# In[6]:


#check for nulls
df.isnull().sum()


# In[7]:


df.head()


# In[8]:


#Make a copy
df2=df.copy()
#the first coulmn is useless
df2.drop(df.columns[0],axis=1)


# In[9]:


# Split it by space for converting it to numeric (since the column means how hotel far from the city center)
df2["Far from city center"]= df2["Are"].str.split(" ", n = 1, expand = False)


# In[10]:


#convert price column from string to float

df2[' price']=df2[' price'].str.replace(',','')
df2[' price']=df2[' price'].astype(float)


# In[11]:


df2["facilities"]= df2["facilities"].str.split(" ", n = 1, expand = False)


# In[12]:


for i in range (df2.shape[0]):
     df2["Far from city center"][i]= (df2["Far from city center"][i][0])
        


# In[13]:


# Convert the (Far from city center) column to float
df2["Far from city center"]= pd.to_numeric( df2["Far from city center"],errors='coerce')
df2["Far from city center"]


# In[14]:


# Because there is some entries in km and some in meter, i ll make it all in km.
for i in range (df2.shape[0]):
     if df2["Far from city center"][i] >100:
            #Selected 100 because no one will write in km if it is above 100!
        df2["Far from city center"][i]= (df2["Far from city center"][i])/1000
        


# # which cities have the most bookings in Saudi Arabia
# 

# In[15]:


plt.hist(df2['city'],color='green')
df2['city'].value_counts()


# In[16]:


mostbooked=df2[ (df2['city']=='Riyadh') | (df2['city']=='jeddah') ]
others=df2[ (df2['city']!='Riyadh') & (df2['city']!='jeddah') ]
#the first coulmn is useless
del mostbooked[mostbooked.columns[0]]
del others[others.columns[0]]
others


# In[17]:


maxpricemost=mostbooked[' price'].max()
averagepricemost=mostbooked[' price'].mean()
minpricemost=mostbooked[' price'].min()
print(maxpricemost)
print(averagepricemost)
print(minpricemost)
maxpriceother=others[' price'].max()
averagepriceother=others[' price'].mean()
minpriceother=others[' price'].min()
print(maxpriceother)
print(averagepriceother)
print(minpriceother)


# In[18]:


plt.figure(figsize=(10,6))
bin_edges = np.arange (minpricemost, mostbooked[' price'].max()+1, 5)
plt.hist(data = mostbooked, x = ' price', bins = bin_edges, color = 'green')
plt.xlim(minpricemost,averagepricemost*2)
plt.xlabel('price in Riyadh and Jeddah')
plt.ylabel('Frequency');


# In[19]:


plt.figure(figsize=(10,6))
bin_edges = np.arange (minpriceother, others[' price'].max()+1, 5)
plt.hist(data = others, x = ' price', bins = bin_edges, color = 'green')
plt.xlim(minpriceother,averagepriceother*2)
plt.xlabel('price in other cities')
plt.ylabel('Frequency');

# The most frequent price in Riyadh and jeddah is around 200 Riyal, and around 250 in other cities# The max and average hotel price is more in other city than Riyadh and Jeddah, in my opnion this because Riyadh and Jeddah have more hotels.
# # Relation between price and other factors

# In[35]:


mostbooked.corr().style.background_gradient(cmap="Greens")


# In[36]:


others.corr().style.background_gradient(cmap="Greens")

There is no big influence of the far from the city center on prices in Riyadh and Jeddah, and vice versa for other cities, maybe that is because of the good services in Riyadh and Jeddah not just in the city center.

# # Is price related to review?
We found no huge influence of review on price for cities except Riyadh and Jeddah, and vice versa for Riyadh and Jeddah, The way I see it, that is because there is price competition in Riyadh and Jeddah according to  they have more hotels.
# In[ ]:





# In[22]:


maxReviewemost=mostbooked[' Review'].max()
averageReviewemost=mostbooked[' Review'].mean()
minReviewmost=mostbooked[' Review'].min()
print(maxReviewemost)
print(averageReviewemost)
print(minReviewmost)
maxReviewother=others[' Review'].max()
averageReviewother=others[' Review'].mean()
minReviewother=others[' Review'].min()
print(maxReviewother)
print(averageReviewother)
print(minReviewother)


# In[23]:


plt.figure(figsize=(10,6))
bin_edges = np.arange (minReviewmost, mostbooked[' Review'].max()+1, 0.2)
plt.hist(data = mostbooked, x = ' Review', bins = bin_edges, color = 'green')
plt.xlim(minReviewmost,maxReviewemost)
plt.xlabel('Review')
plt.ylabel('Frequency');


# In[25]:


plt.figure(figsize=(10,6))
bin_edges = np.arange (minReviewmost, others[' Review'].max()+1, 0.2)
plt.hist(data = others, x = ' Review', bins = bin_edges, color = 'green')
plt.xlim(minReviewother,maxReviewother)
plt.xlabel('Review')
plt.ylabel('Frequency');


# # What is the range of prices in saudi booking?

# In[37]:


fig = plt.figure(figsize =(10, 7))
 
plt.boxplot(df2[' price'] )
 
plt.show()

# 75% of hotel price in Saudi arabia is under 1000 Riyal
# In[ ]:





# In[ ]:





# In[28]:


#High price hotel is the hotel with price aboce 1000 Riyal (far from the box plot).
highprice=df2[df2[' price'] > 1000]
lowprice=df2[df2[' price'] < 1000]


# In[29]:


# How many?
print(highprice.shape[0])#the first coulmn is useless

# Percentage?
print(highprice.shape[0]/df2.shape[0] * 100)


# In[30]:


fig = plt.figure(figsize =(10, 7))
 
plt.boxplot(highprice[' Review'] )
 
plt.show()

# 75% of high hotel price review in other cities is between 7.75 and ≈8.3 out of 10
# In[ ]:





# # What is the difference between the most frequent facilities in high price and low price hotel?

# In[46]:


print(highprice['facilities'].value_counts())
10/86 * 100


# In[47]:


print(lowprice['facilities'].value_counts())
lowprice['facilities'].shape[0]


# In[ ]:




