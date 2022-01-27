import pandas as pd
import numpy as np
#---------Plots---------
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
#--------Models-------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#--------Reporting--------
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#Restructuring Dataset and Labeling Features

df = pd.read_csv("dataClustering.csv",header =None)
for i in range(0,8):
    df = df.rename(columns = {i:f'X{i+1}'})

print(df)

#Initial Visualizations, Seems to have 3-4 distinct clustering on the pairplot
sns.pairplot(df)

#I chose K-Means Clustering to find distinctive clusters, the intertia drastically drops
#with dimishing decresing distances at 4

X = df.to_numpy()
len(X)
y=[]
K = range(1,15)
for n in K:
    kmeans = sklearn.cluster.KMeans(n_clusters=n).fit(X)
    y.append(kmeans.inertia_)

fig, ax  = plt.subplots(figsize = (10,10))
ax.plot(K,y, color = 'blue',label = '')

ax.legend(frameon = False)
ax.set_xlabel('k',size=14)
ax.set_ylabel('Inertia',size=14)
ax.set_title('Kmeans Cluster Optimization',size=20)
plt.show()



#Fitting Kmeans and choosing 4 clusters
kmeans = sklearn.cluster.KMeans(n_clusters=4, random_state=0).fit(X)
label = pd.DataFrame(kmeans.labels_)

df_lab = pd.concat((df,label),axis = 1)
df_lab = df_lab.rename(columns = {0:'class'})

print(df_lab['class'].unique())
print(df_lab)

#redoing pairplot on labeled dataset
sns.pairplot(df_lab,hue='class')

#Train test split for classification model

X_train, X_test, y_train, y_test = train_test_split(
                                                    df_lab.drop('class',axis=1),
                                                    df_lab['class'],
                                                    test_size=0.30)

#Multinomial Regression Fitting
logit = LogisticRegression(solver='liblinear')
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)


print(f'The Logistic model correctly predicts the label {100*metrics.accuracy_score(y_test, y_pred):.2f}% of the time.')

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9,9))

sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax,annot=True, annot_kws={"size": 16}) #

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Logit Model Confusion Matrix', fontsize=18)
plt.show()

#Trying SVM
from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print(f'The SVM model correctly predicts {100*metrics.accuracy_score(y_test, y_pred):.2f}% of the time.')

#Random Forest

from sklearn.ensemble import RandomForestClassifier as rf

clf_r = rf(n_estimators = 10)
clf_r.fit(X_train, y_train)
y_pred = clf_r.predict(X_test)


print(f'The Random Forest model correctly predicts {100*metrics.accuracy_score(y_test, y_pred):.2f}% of the time.')


#All models perform at 100%, but what if there are more than 4 actual labels?

#Create new clusters, refit models

#Fitting Kmeans and choosing 10 clusters

mlm = []
sv = []
randf = []
clust = range(4,20)

for i in clust:
    X_train, X_test, y_train, y_test = train_test_split(
                                                    df_lab.drop('class',axis=1),
                                                    df_lab['class'],
                                                    test_size=0.30)
    #clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=0).fit(X)
    label = pd.DataFrame(kmeans.labels_)
    df_lab = pd.concat((df,label),axis = 1)
    df_lab = df_lab.rename(columns = {0:'class'})
    #-----------------------------------------------
    #Logit
    logit = LogisticRegression(solver='liblinear')
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    
    mlm.append(100*metrics.accuracy_score(y_test, y_pred))
    #-------------------------------------------------
    #SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    sv.append(100*metrics.accuracy_score(y_test, y_pred))
    #--------------------------------------------------
    #Random Forest
    clf_r = rf(n_estimators = 10)
    clf_r.fit(X_train, y_train)
    y_pred = clf_r.predict(X_test)
    randf.append(100*metrics.accuracy_score(y_test, y_pred))


    
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(clust,mlm, color = 'blue',label = 'Logit')
ax.plot(clust,sv, color = 'red',label = 'SVM')
ax.plot(clust,randf, color = 'green',label = 'Random Forest')
ax.legend(frameon = False)
ax.set_xlabel('K-Cluster Size',size=14)
ax.set_ylabel('Accuracy',size=14)
ax.set_title('Prediction Accuracy by Cluster Size',size=20)
plt.show()

