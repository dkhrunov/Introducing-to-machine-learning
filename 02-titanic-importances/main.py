#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[247]:


data = pd.read_csv("titanic.csv", index_col="PassengerId")
data.head(10)


# In[248]:


X_labels = ['Pclass','Fare','Age','Sex']
X = data.loc[:,X_labels]


# In[249]:


X['Sex'] = X['Sex'].map(lambda sex: 1 if sex == 'male' else 0)


# In[250]:


y = data['Survived']


# In[251]:


X = X.dropna(axis=0)
y = y[X.index.values]


# In[252]:


clf = DecisionTreeClassifier(random_state=241)
clf.fit(X.values, y.values)


# In[253]:


arr = {name: score for name, score in zip(X.columns, clf.feature_importances_)}
arr


# In[254]:


v = np.array(list(arr.values()))
k = np.array(list(arr.keys()))

i = np.where(v == v.max())
imp1 = k[i][0]
v = np.delete(v, i)
k = np.delete(k, i)


j = np.where(v == v.max())
imp2 = k[j][0]
v = np.delete(v, j)
k = np.delete(k, i)

print(imp1, imp2)


# In[ ]:




