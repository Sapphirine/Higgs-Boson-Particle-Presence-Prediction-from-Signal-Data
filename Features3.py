
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


feature_text='label, lepton pT, lepton eta, lepton phi, lost energy mag, lost energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[a.strip() for a in feature_text.split(',')]

dataset = pd.read_csv('newhiggs.csv', names=features)


# In[25]:


from sklearn.model_selection import train_test_split
f = features[1:]

x_train, x_test, y_train, y_test = train_test_split(dataset[f], dataset['label'], stratify=dataset['label'])


# In[43]:


x_train.plot(kind='density', subplots=True, layout=(4,7), figsize = (30,10), sharex=False)
plt.show()


# In[20]:


a = ['label','jet 3 b-tag', 'jet 1 b-tag', 'm_lv', 'm_jj', 'm_jjj', 'lost energy phi', 'm_bb']
custom_map = {0: 'boson absent', 1: 'boson present'}
dataset['label'] = dataset['label'].map(custom_map)

#x_train[a]


# In[21]:


g = sns.pairplot(dataset[a], hue='label')


# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[26]:


clf = RandomForestClassifier()
param_grid = dict(max_depth=[1, 2, 5, 10, 20, 30, 40],
                  min_samples_split=[2, 5, 10],
                  min_samples_leaf=[2, 3, 5])
est = GridSearchCV(clf, param_grid=param_grid, n_jobs=8)


# In[27]:


est.fit(x_train, y_train)


# In[28]:


scores = pd.DataFrame(est.cv_results_)
scores.head()


# In[29]:


sns.factorplot(x='param_max_depth', y='mean_test_score',
               col='param_min_samples_split',
               hue='param_min_samples_leaf',
               data=scores);


# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[33]:


clf = DecisionTreeClassifier()
param_grid = dict(max_depth=[1, 2, 5, 10, 20, 30, 40],
                  min_samples_split=[2, 5, 10],
                  min_samples_leaf=[2, 3, 5])


# In[34]:


est = GridSearchCV(clf, param_grid=param_grid, n_jobs=8)


# In[35]:


est.fit(x_train, y_train)


# In[36]:


scores = pd.DataFrame(est.cv_results_)


# In[37]:


sns.factorplot(x='param_max_depth', y='mean_test_score',
               col='param_min_samples_split',
               hue='param_min_samples_leaf',
               data=scores);


# In[38]:


from sklearn.naive_bayes import GaussianNB


# In[39]:


clf = GaussianNB()


# In[40]:


get_ipython().run_line_magic('pinfo', 'GaussianNB')

