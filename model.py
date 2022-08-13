#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import cv2
import glob
import os
import matplotlib.pyplot as plt
import string
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


for dirname, _, filenames in os.walk('../PRML project/fruits-360_'):
    for filename in filenames[:2]:
        print(os.path.join(dirname, filename))


# # For getting Training Dataset

# In[3]:


fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("../PRML project/fruits-360_/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)


# In[4]:


labels #gettting address of all fruit subfolders in an array


# In[5]:


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

label_to_id_dict 


# In[6]:


id_to_label_dict #label encoding of all fruits


# In[7]:


label_ids = np.array([label_to_id_dict[x] for x in labels])
np.unique(label_ids)


# # For getting Validation Dataset

# In[8]:


fruit_imagesT = []
labelsT = [] 
for fruit_dir_path in glob.glob("../PRML project/fruits-360_/Validation/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_imagesT.append(image)
        labelsT.append(fruit_label)
fruit_imagesT = np.array(fruit_imagesT)
labelsT = np.array(labelsT)


# In[9]:


label_to_id_dictT = {v:i for i,v in enumerate(np.unique(labelsT))}
id_to_label_dictT = {v: k for k, v in label_to_id_dictT.items()}

id_to_label_dictT


# In[10]:


label_idsT = np.array([label_to_id_dictT[x] for x in labelsT])
np.unique(label_idsT)


# In[ ]:





# # Standardising Training and Testing Dataset

# In[11]:


scaler = StandardScaler()
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])
images_scaledT = scaler.transform([i.flatten() for i in fruit_imagesT])


# In[12]:


len(images_scaled)


# In[13]:


images_scaled


# In[ ]:





# # Finding best value of number of components for applying PCA on Training  Dataset

# In[ ]:


pca = PCA().fit(images_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim([0, 150])
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')


# After Observing the graph we see 120 components capture approx 90% variance

# # Applying PCA on the Datasets with number of components=120

# In[14]:


pca = PCA(n_components=120)
X = pca.fit_transform(images_scaled)


# In[15]:


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, label_ids, test_size=0.30, random_state=42)


# In[16]:


X_val1=pca.transform(images_scaledT)
Y_val1=label_idsT


# In[17]:


len(Y_train1)


# In[18]:


len(Y_test1)


# In[19]:


len(Y_val1)


# In[ ]:





# # Model 1:Random Forest Classifier

# In[20]:


rfc = RandomForestClassifier(max_depth=48, random_state=24)
rfc=rfc.fit(X_train1, Y_train1)


# In[21]:


y_pred = rfc.predict(X_test1)


# In[22]:


precision = metrics.accuracy_score(y_pred, Y_test1) * 100
print("Accuracy with Random Forest Classifier: {0:.2f}%".format(precision))


# Accuracy on Validation set¶

# In[23]:


Y_pred2 = rfc.predict(X_val1)


# In[24]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with Random Forest Classifier: {0:.6f}".format(precision2))


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[26]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# # Model 2:Logistic Regression

# In[27]:


log = LogisticRegression()
log=log.fit(X_train1, Y_train1)


# In[28]:


y_pred = log.predict(X_test1)


# In[29]:


precision = metrics.accuracy_score(y_pred, Y_test1) * 100
print("Accuracy with Logistic: {0:.2f}%".format(precision))


# Accuracy on Validation set¶

# In[30]:


Y_pred2 = log.predict(X_val1)


# In[31]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with Logistic Regression: {0:.2f}".format(precision2))


# In[32]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[33]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# # Model 3:KNeighbors Classifier

# In[79]:


from sklearn.neighbors import KNeighborsClassifier 
error =[]
for i in range(1,60):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_train1,Y_train1)
    pred_i=knn2.predict(X_test1)
    error.append(np.mean(pred_i!=Y_test1))        


# In[80]:


plt.figure(figsize=(12,8))
plt.plot(range(1, 60), error, color ='r', linestyle='dashed',marker='o',
         markerfacecolor='blue',markersize=10)
plt.title('Error rate for K value')
plt.xlabel("Value of K")
plt.ylabel("Mean Error")
plt.show()


# Looking at the graph we get the lowest error for n_neighbors=2

# In[35]:


knn = KNeighborsClassifier(n_neighbors=2)
knn=knn.fit(X_train1, Y_train1)


# In[36]:


y_pred = knn.predict(X_test1)


# In[37]:


precision = metrics.accuracy_score(y_pred, Y_test1) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))


# Accuracy on Validation set¶

# In[38]:


Y_pred2 = knn.predict(X_val1)


# In[39]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with KNN: {0:.6f}".format(precision2))


# In[40]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[41]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# # Model 4:MLP

# In[42]:


mlp = MLPClassifier()
mlp=mlp.fit(X_train1, Y_train1)


# In[43]:


y_pred = mlp.predict(X_test1)


# In[44]:


precision = metrics.accuracy_score(y_pred, Y_test1) * 100
print("Accuracy with MLP: {0:.2f}%".format(precision))


# Accuracy on Validation set¶

# In[45]:


Y_pred2 = mlp.predict(X_val1)


# In[46]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with MLP: {0:.6f}".format(precision2))


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[48]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# # Model 5: Bernoulli Naive Bayes  

# In[56]:


gnb = GaussianNB()
gnb=gnb.fit(X_train1, Y_train1)


# In[57]:


y_pred = gnb.predict(X_test1)


# In[58]:


precision=metrics.accuracy_score(y_pred,Y_test1)*100
print("Accuracy with Gaussian Naive Bayes: {0:.6f}".format(precision))


# Accuracy on Validation Dataset

# In[59]:


Y_pred2 = gnb.predict(X_val1)


# In[60]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with Gaussian Naive Bayes: {0:.6f}".format(precision2))


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[62]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# # Model 6:SVM with "linear" kernel

# In[49]:


svm=svm.SVC(kernel='linear',gamma='auto')
svm=svm.fit(X_train1,Y_train1)


# In[50]:


y_pred= svm.predict(X_test1)


# In[51]:


precision=metrics.accuracy_score(y_pred,Y_test1)*100
print("Accuracy with SVM: {0:.6f}".format(precision))


# Accuracy on Validation Dataset

# In[52]:


Y_pred2 = svm.predict(X_val1)


# In[53]:


precision2 = accuracy_score(Y_pred2, Y_val1) * 100
print("Accuracy with SVM: {0:.6f}".format(precision2))


# In[54]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_val1, Y_pred2))


# In[55]:


print(confusion_matrix(Y_val1, Y_pred2))


# In[ ]:





# In[ ]:




