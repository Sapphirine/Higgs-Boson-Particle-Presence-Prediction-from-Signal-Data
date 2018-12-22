
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np 
import matplotlib.pyplot as plt 
import pyspark 
from pyspark.sql import SQLContext


conf = pyspark.SparkConf().setAll([('spark.executor.memory', '5g'), ('spark.driver.maxResultSize', '12g'), ('spark.driver.memory','38g')]) 
sc = pyspark.SparkContext(conf=conf) 
sqlContext = SQLContext(sc)


# In[97]:


import time
from time import time
#from plot_utils import *
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees 
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# In[7]:


feature_text='lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[a.strip() for a in feature_text.split(',')]


# In[8]:


inputRDD=sc.textFile('newhiggs.csv') #Replace with actual path
inputRDD.first()


# In[9]:


Data=(inputRDD.map(lambda line: [float(x.strip()) for x in line.split(',')]).map(lambda line: LabeledPoint(line[0], line[1:])))
Data.first()


# In[51]:


import pyspark
import os

#craete spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('higgs-boson-detection').getOrCreate()

#read dataset from local filesystem
data_location = os.path.join('','newhiggs.csv')
df = spark.read.load(data_location, format="csv", sep=",", inferSchema="true", header="true")
#df.show(10)
df.head(20)


# In[52]:


(training, test) = df.randomSplit([0.7, 0.3])
training.count(), test.count()


# In[53]:


from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector

training_dense  = training.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
training_dense = spark.createDataFrame(training_dense, ["label", "features"])

test_dense = test.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
test_dense = spark.createDataFrame(test_dense, ["label", "features"])


# In[54]:


#GBT Model
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg


def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Type not supported {0}".format(type(v)))

labelPoint_train = training_dense.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
labelPoint_train.take(2)


# In[100]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=10)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[101]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=15)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[102]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=20)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[103]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=25)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[104]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=30)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[105]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=35)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[106]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=40)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[108]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=45)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[109]:


start = time.time()
GBTmodel = GradientBoostedTrees.trainClassifier(labelPoint_train,categoricalFeaturesInfo={}, numIterations=50)
end = time.time()
print(f'Time taken to train model using GBT: {end - start} seconds')

predictions = GBTmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[110]:


x = [10,15,20,25,30,35,40,45,50]
y = [0.3283272569154096,0.3169548784158946,0.31167006723082585,0.3049135364752316,0.30477974378700207,0.3006990667959996,0.29795631668729305,0.2974545941064321,0.2951801184065291]
plt.plot(x,y, color='maroon')
plt.xlabel('numIterations')
plt.ylabel('Test Error')
plt.title('Hyperparameter Tuning - GBT ')
plt.show()


# In[61]:


from pyspark.mllib.tree import RandomForest, RandomForestModel

start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')

predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[80]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=1, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[65]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=2, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[66]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=3, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[67]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[68]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=5, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[75]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[72]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=7, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[77]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=8, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[76]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=9, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[78]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=10, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[114]:


x = [1,2,3,4,5,6,7,8,9,10]
y = [0.41713215372779877,0.382948121885139,0.37184332876208315,0.3381944676723417,0.3302338027226812,0.3151821252968525,0.3086597317456601,0.30347526507676353,0.29514667023447166,0.2877546242097869]
plt.plot(x,y, color='maroon')
plt.xlabel('maxDepth')
plt.ylabel('Test Error')
plt.title('Hyperparameter Tuning - Random Forest')
plt.show()


# In[87]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[88]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=20, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[89]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=30, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[92]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=35, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[91]:


start = time.time()
RFmodel = RandomForest.trainClassifier(labelPoint_train,
                                     numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=40, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=6, maxBins=32)
end = time.time()
print(f'Time taken to train model using RF: {end - start} seconds')
predictions  = RFmodel.predict(test_dense.rdd.map(lambda x: x.features.values))
labelsAndPredictions = test_dense.rdd.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_dense.rdd.count())
print('Test Error = ' + str(testErr))


# In[ ]:


x = [1,2,3,4,5,6,7,8,9,10]
y = [0.41713215372779877,0.382948121885139,0.37184332876208315,0.3381944676723417,0.3302338027226812,0.3151821252968525,0.3086597317456601,0.30347526507676353,0.29514667023447166,0.2877546242097869]
plt.plot(x,y, color='orange')
plt.xlabel('maxDepth')
plt.ylabel('Test Error')
plt.title('Hyperparameter Tuning - Random Forest')
plt.show()

