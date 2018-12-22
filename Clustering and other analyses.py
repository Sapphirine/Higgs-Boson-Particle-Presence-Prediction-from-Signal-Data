
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SQLContext
# sc.stop()
# conf = pyspark.SparkConf().setAll([('spark.executor.memory', '5g'), ('spark.driver.maxResultSize', '12g'), ('spark.driver.memory','38g')])
conf = pyspark.SparkConf().setAll([('spark.driver.maxResultSize', '12g'), ('spark.driver.memory','38g')])
sc = pyspark.SparkContext(conf=conf)
sqlContext = SQLContext(sc)


# In[2]:


from time import time
#from plot_utils import *
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees 
from pyspark.mllib.tree import GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# In[ ]:


feature_text='lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[a.strip() for a in feature_text.split(',')]


# In[ ]:


inputRDD=sc.textFile('HIGGS.csv') #Replace with actual path
inputRDD.first()


# In[ ]:


Data = (inputRDD.map(lambda line: [float(x.strip()) for x in line.split(',')]).map(lambda line: LabeledPoint(line[0], line[1:])))
Data.first()


# In[ ]:


Data1=Data.sample(False,0.01).cache()
(trainingData,testData) = Data1.randomSplit([0.7,0.3])

print ('Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data1.count(),trainingData.cache().count(),testData.cache().count()))


# In[ ]:


type(Data1)


# In[ ]:


errors={} 
for depth in [1,3,6,10]:     
    start=time()        
    model=GradientBoostedTrees.trainClassifier(Data1,categoricalFeaturesInfo={}, numIterations=3)
    errors[depth]={}     
    dataSets={'train':trainingData,'test':testData}     
    for name in dataSets.keys():  
        # Calculate errors on train and test sets           
        data=dataSets[name]         
        Predicted=model.predict(data.map(lambda x: x.features))         
        LabelsAndPredictions=(data.map(lambda lp: lp.label).zip(Predicted))
        Err=(LabelsAndPredictions.filter(lambda v:v[0] != v[1]).count()/float(data.count()))         
        errors[depth][name]=Err     
    print(depth,errors[depth],int(time()-start),'seconds')
print(errors)


# In[ ]:


errors={}
for depth in [1,3,6,10,15,20]:
    start=time()
    model = RandomForest.trainClassifier(Data1, numClasses=2, 
           categoricalFeaturesInfo={}, numTrees=3,  
           featureSubsetStrategy="auto", impurity='gini', 
           maxDepth=4, maxBins=32)
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():
        # Calculate errors on train and test sets
        data = dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=(data.map(lambda lp: lp.label).zip(Predicted))
        Err=(LabelsAndPredictions.filter(lambda v:v[0] != v[1]).count()/float(data.count()))
        errors[depth][name]=Err
    print(depth,errors[depth],int(time()-start),'seconds')
print(errors)


# In[ ]:


## Clustering Starts - Siddhant


# In[ ]:


inputRDD=sc.textFile('HIGGS.csv') #Replace with actual path
inputRDD.first()


# In[ ]:


Data=(inputRDD.map(lambda line: [float(x.strip()) for x in line.split(',')]).map(lambda line: LabeledPoint(line[0], line[1:])))
Data.first()
clusteringData=Data.sample(False,0.01).cache()
(clusterTrain,clusterTest)=clusteringData.randomSplit([0.7,0.3])


# In[ ]:


cTrain = clusterTrain.collect()
cTest = clusterTest.collect()


# In[ ]:


clusterTrain.collect()[0]


# In[ ]:


trainFeatures = clusterTrain.map(lambda lp: lp.features).collect()


# In[ ]:


trainLabels = clusterTrain.map(lambda lp: lp.label).collect()


# In[ ]:


trainLabels[1]


# In[ ]:


trainDF = [x.toArray() for x in trainFeatures]
# trainLabelDF = [x.toArray() for x in trainLabels]
len(trainDF)
# len(trainLabelDF)


# In[ ]:


trainDF[0]


# In[ ]:


trainFeatures[0].toArray()


# In[ ]:


clusteringData.collect()[0]


# In[ ]:


# from pyspark.ml.clustering import KMeans
from sklearn.cluster import KMeans
# dataset = clusterTrain
dataset = trainDF
# Trains a k-means model.
# kmeans = KMeans().setK(2).setSeed(122)
kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)
model = kmeans.fit(dataset)
# model = kmeans.fit(dataset)
# Evaluate clustering by computing Within Set Sum of Squared Errors.
# wssse = model.computeCost(dataset)
# print("Within Set Sum of Squared Errors = " + str(wssse))
# Shows the result.
# centers = model.clusterCenters()
# print("Cluster Centers: ")
# for center in centers:
# print(center)


# In[ ]:


from sklearn.metrics.cluster import completeness_score
transformed = model.predict(dataset)
# labels = dataset.select('Label').collect()
labels = trainLabels
label_array = [int(i) for i in labels]
preds = transformed
preds_array = [int(i) for i in preds]
completeness_score(preds_array, label_array)


# In[ ]:


trainDF[0]


# In[ ]:


from sklearn.cluster import SpectralClustering
# dataset = clusterTrain
dataset = trainDF
# Trains a k-means model.
# kmeans = KMeans().setK(2).setSeed(122)
spectral = SpectralClustering(n_clusters=2, random_state=23).fit(dataset)


# In[ ]:


## Clustering Ends - Siddhant


# In[3]:


df = sqlContext.read.format('com.databricks.spark.csv')     .options(header='false', inferschema='true')     .load('newhiggs.csv')


# In[4]:


df.show(n=3)


# In[5]:


df.columns


# In[6]:


featuredf = df.select([c for c in df.columns if c not in {'_c0'}])
labeldf = df.select('_c0')


# In[7]:


featuredf.columns


# In[80]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=featuredf.columns,
    outputCol="features")

outputFeatureDf = assembler.transform(featuredf)
# output.select("name", "marks").show(truncate=False)


# In[77]:


outputFeatureDf.show(n=2)


# In[85]:


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
fDf = assembler.transform(featuredf)
model = pca.fit(fDf)
result = model.transform(fDf).select("pcaFeatures")


# In[86]:


result.show(n=1)


# In[87]:


from pyspark.ml.clustering import KMeans
dataset = outputFeatureDf
# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(122)
model = kmeans.fit(dataset)
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[88]:


from sklearn.metrics.cluster import completeness_score
transformed = model.transform(dataset)
labels = labeldf.collect()
label_array = [int(i[0]) for i in labels]
preds = transformed.select('prediction').collect()
preds_array = [int(i.prediction) for i in preds]
completeness_score(preds_array,label_array)


# In[90]:


ss = result.collect()


# In[93]:


xPlot = [x.pcaFeatures[0] for x in ss]
yPlot = [x.pcaFeatures[1] for x in ss]


# In[92]:


ss[0].pcaFeatures[0]


# In[95]:


preds_array


# In[96]:


newPreds = []
plt.scatter(xPlot,yPlot,c=preds_array)


# In[35]:


dataset = outputFeatureDf
kValues = [2,3,4,5,6,7,8]
wssse = []
for k in kValues:
    kmeans = KMeans().setK(k).setSeed(122)
    model = kmeans.fit(dataset)
    wssse.append(model.computeCost(dataset))
for i in wssse:
    print(i)


# In[29]:


from pyspark.ml.clustering import BisectingKMeans
# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1222)
model = bkm.fit(outputFeatureDf)
# Evaluate clustering.
cost = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(cost))
# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)


# In[30]:


from sklearn.metrics.cluster import completeness_score
transformed = model.transform(dataset)
labels = labeldf.collect()
label_array = [int(i[0]) for i in labels]
preds = transformed.select('prediction').collect()
preds_array = [int(i.prediction) for i in preds]
completeness_score(preds_array,label_array)


# In[36]:


dataset = outputFeatureDf
kValues = [2,3,4,5,6,7,8]
bwssse = []
for k in kValues:
    bkmeans = BisectingKMeans().setK(k).setSeed(122)
    bmodel = bkmeans.fit(dataset)
    bwssse.append(bmodel.computeCost(dataset))
for i in bwssse:
    print(i)


# In[31]:


from pyspark.ml.clustering import GaussianMixture
gmm = GaussianMixture(predictionCol="prediction").setK(2).setSeed(538009335)
gmmmodel = gmm.fit(outputFeatureDf)
print("Gaussians shown as a DataFrame: ")
gmmmodel.gaussiansDF.show()


# In[32]:


from sklearn.metrics.cluster import completeness_score
transformed = gmmmodel.transform(dataset)
labels = labeldf.collect()
label_array = [int(i[0]) for i in labels]
preds = transformed.select('prediction').collect()
preds_array = [int(i.prediction) for i in preds]
completeness_score(preds_array,label_array)


# In[51]:


dataset = outputFeatureDf
kValues = [2,3,4,5,6,7,8]
gmmError = []
for k in kValues:
    gmm = GaussianMixture(predictionCol="prediction").setK(k).setSeed(538009335)
    gmmmodel = gmm.fit(dataset)
    transformed = gmmmodel.transform(dataset)
    labels = labeldf.collect()
    label_array = [int(i[0]) for i in labels]
    preds = transformed.select('prediction').collect()
    preds_array = [int(i.prediction) for i in preds]
    gmmError.append(completeness_score(preds_array,label_array))
for i in gmmError:
    print(i)


# In[48]:


plt.plot(kValues,wssse, color="violet")
plt.xlabel('Number of Clusters')
plt.ylabel('Within sum of squared errors')
plt.title('Hyperparameter Tuning - K means Clustering')
plt.show()


# In[45]:


plt.plot(kValues,bwssse, color="purple")
plt.xlabel('Number of Clusters')
plt.ylabel('Within sum of squared errors')
plt.title('Hyperparameter Tuning - BK means Clustering')
plt.show()


# In[52]:


plt.plot(kValues,gmmError, color="black")
plt.xlabel('Number of bayesians')
plt.ylabel('Completeness Score')
plt.title('Hyperparameter Tuning - GMM')
plt.show()


# In[73]:


# inputs
# In [41]: num = np.array([1, 2, 3, 4, 5])
# In [42]: sqr = np.array([1, 4, 9, 16, 25])

# convert to pandas dataframe
d = {'techs': techs, 'accuracy': acc}
pdnumsqr = pd.DataFrame(d)

