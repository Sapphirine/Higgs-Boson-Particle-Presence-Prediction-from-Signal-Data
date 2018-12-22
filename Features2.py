
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


feature_text='label, lepton pT, lepton eta, lepton phi, lost energy mag, lost energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[a.strip() for a in feature_text.split(',')]

dataset = pd.read_csv('newhiggs.csv', names=features)


# In[3]:


from sklearn.model_selection import train_test_split
f = features[1:]
x_train, x_test, y_train, y_test = train_test_split(dataset[f], dataset['label'], stratify=dataset['label'])


# In[4]:


x_train.plot(kind='density', subplots=True, layout=(7,4), figsize = (20,20), sharex=False)
plt.show()


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer 

from sklearn.metrics import f1_score ,average_precision_score ,roc_auc_score


logreg1 = LogisticRegression()
logreg1.fit(x_train,y_train)
logreg1.score(x_test,y_test)


# In[6]:


print("Test Avg Precision score: ", average_precision_score(logreg1.predict(x_test),y_test))
print("Test F1 score: ", f1_score(logreg1.predict(x_test),y_test))
print("Test ROC AUC score: ", roc_auc_score(logreg1.predict(x_test),y_test))


# In[7]:


logreg1.coef_[0]


# In[8]:


colour = []
for i in range(14):
    if i < 7:
        colour.append("red")
    else:
        colour.append("blue")

def plot(coef,feature_names,i): 
    top20_index_pos = coef.argsort()[-7:] 
    top20_pos = coef[top20_index_pos]
    print (top20_pos)
    top20_names_pos = [feature_names[j] for j in top20_index_pos] 
    print(top20_names_pos)
    top20_index_neg = coef.argsort()[:7] 
    top20_neg = coef[top20_index_neg]
    print (top20_neg)
    top20_names_neg = [feature_names[j] for j in top20_index_neg] 
    print(top20_names_neg)
    top_coef = np.hstack([top20_neg,top20_pos])
    print(top_coef)
    top_names = np.hstack([top20_names_neg,top20_names_pos])
    print(top_names)
    plt.figure(figsize=(10,4))
    plt.bar(range(1,15),top_coef,color=colour)
    plt.title('most important features '+str(i))
    plt.xticks(range(1,15),top_names,rotation=45)
    plt.show()


# In[67]:


plot(logreg1.coef_[0], f, 1)


# In[10]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# In[11]:


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


# In[12]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=20),
    RandomForestClassifier(max_depth=20, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# In[13]:


scaler = StandardScaler()
scaler.fit(x_train)


# In[14]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[15]:


logreg1 = LogisticRegression()
logreg1.fit(x_train,y_train)
logreg1.score(x_test,y_test)


# In[16]:


print("Test Avg Precision score: ", average_precision_score(logreg1.predict(x_test),y_test))
print("Test F1 score: ", f1_score(logreg1.predict(x_test),y_test))
print("Test ROC AUC score: ", roc_auc_score(logreg1.predict(x_test),y_test))


# In[17]:


all_clf = []
for clf, n in zip(classifiers, names):
    clf.fit(x_train, y_train)
    k = clf.score(x_test, y_test)
    print(n, ': ', k)
    all_clf.append(clf)


# In[19]:


new = ["Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
c = [    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# In[20]:


new_clf = []
for clf, n in zip(c, new):
    clf.fit(x_train, y_train)
    k = clf.score(x_test, y_test)
    print(n, ': ', k)
    all_clf.append(clf)


# In[35]:





# In[62]:


dict1 = {'RBF SVM':0.53096,'Naive Bayes': 0.60264,'k Nearest Neighbours': 0.60948, 'Logistic Regression':0.63512, 'Linear SVM': 0.63768,'Decision Tree': 0.6801,'Multi Layer Perceptron': 0.69008,
       'Random Forest': 0.6925, 'Gradient Boosting': 0.7 
       }


# In[63]:


g = sns.barplot(list(dict1.keys()), list(dict1.values()))
#g.set_xticklabels(rotation=30)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('Accuracy')

