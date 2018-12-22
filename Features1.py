
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SQLContext

conf = pyspark.SparkConf().setAll([('spark.executor.memory', '5g'), ('spark.driver.maxResultSize', '12g'), ('spark.driver.memory','38g')])
sc = pyspark.SparkContext(conf=conf)
sqlContext = SQLContext(sc)


# In[2]:


from time import time
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees 
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# In[3]:


feature_text='lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[a.strip() for a in feature_text.split(',')]


# In[4]:


inputRDD=sc.textFile('HIGGS.csv') #Replace with actual path
inputRDD.first()


# In[5]:


Data=(inputRDD.map(lambda line: [float(x.strip()) for x in line.split(',')]).map(lambda line: LabeledPoint(line[0], line[1:])))
Data.first()


# In[6]:


Data1=Data.sample(False,0.01).cache()
(trainingData,testData)=Data1.randomSplit([0.7,0.3])
print ('Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data1.count(),trainingData.cache().count(),testData.cache().count()))


# In[9]:


from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.ml.classification import LogisticRegression


# In[10]:


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[12]:


from pyspark.mllib.classification import LogisticRegressionWithSGD


# In[15]:


lr = LogisticRegressionWithSGD.train(Data1)


# In[31]:


lr.weights


# In[22]:


plt.plot(np.sort(lr.weights.toArray()))


# In[23]:


from pyspark.mllib.stat import Statistics


# In[ ]:


features = df.rdd.map(lambda row: row[1:])
corr_mat=Statistics.corr(features, method="pearson")

