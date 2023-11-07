#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#classifiers
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#model evaluation
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#data processing function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Problem Statement : Loan Approval Problem
#It is binary classfication problem 


# In[12]:


data=pd.read_csv("loan_prediction.csv")
data.sample(25)


# In[11]:


data.shape


# In[14]:


data.dtypes


# In[16]:


sns.countplot(x="Gender",hue="Loan_Status",data=data)


# In[17]:


sns.countplot(x="Married",hue="Loan_Status",data=data)


# In[18]:


correlation_mat = data.corr()


# In[19]:


sns.heatmap(correlation_mat,annot=True,linewidths=5,cmap="YlGnBu")


# In[21]:


sns.pairplot(data)#there is +ve relation between applicantincome


# In[22]:


data.isnull().sum()


# In[26]:


data["Gender"].fillna(data["Gender"].mode()[0],inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0],inplace=True)
data["Married"].fillna(data["Married"].mode()[0],inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0],inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0],inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0],inplace=True)


#all values of dependents column were of str form converting to int form
data["Dependents"] = data["Dependents"].replace('3+',int(3))
data["Dependents"] = data["Dependents"].replace('1',int(1))
data["Dependents"] = data["Dependents"].replace('2',int(2))
data["Dependents"] = data["Dependents"].replace('0',int(0))

data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
print(data.isnull().sum())













# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[37]:


data["Gender"] = le.fit_transform(data["Gender"])
data["Married"] = le.fit_transform(data["Married"])
data["Education"] = le.fit_transform(data["Education"])
data["Self_Employed"] = le.fit_transform(data["Self_Employed"])
data["Property_Area"] = le.fit_transform(data["Property_Area"])
data["Loan_Status"] = le.fit_transform(data["Loan_Status"])

data.head(5)


# In[32]:


x=data.drop(["Loan_Status","Loan_ID"],axis=1)
y=data["Loan_Status"]


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[35]:


model=LogisticRegression(solver="liblinear")


# In[36]:


model.fit(x_train,y_train)


# In[38]:


model.score(x_train,y_train)


# In[40]:


model.score(x_test,y_test)


# In[41]:


dtree =  DecisionTreeClassifier(criterion="gini")
dtree.fit(x_train,y_train)


# In[42]:


dtree.score(x_train,y_train)


# In[43]:


dtree.score(x_test,y_test)


# In[47]:


dtreeR =  DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=0)
dtreeR.fit(x_train,y_train)


# In[48]:


dtreeR.score(x_train,y_train)


# In[49]:


dtreeR.score(x_test,y_test)


# In[50]:


from sklearn import metrics


# In[53]:


from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier(n_estimators=150,base_estimator=dtreeR,random_state=0)
bgcl.fit(x_train,y_train)


# In[54]:


bgcl.score(x_train,y_train)


# In[55]:


bgcl.score(x_test,y_test)


# In[56]:


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(x_train,y_train)


# In[57]:


model_lr.score(x_train,y_train)


# In[58]:


model_lr.score(x_test,y_test)


# In[81]:


from sklearn.ensemble import AdaBoostClassifier
abcl=AdaBoostClassifier(n_estimators=90,random_state=0)
abcl.fit(x_train,y_train)


# In[82]:


abcl.score(x_train,y_train)


# In[83]:


abcl.score(x_test,y_test)


# In[101]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl=GradientBoostingClassifier(n_estimators=60,random_state=0)
gbcl.fit(x_train,y_train)


# In[102]:


gbcl.score(x_train,y_train)


# In[161]:


gbcl.score(x_test,y_test)


# In[171]:


from sklearn.ensemble import RandomForestClassifier
rfcl=RandomForestClassifier(max_depth=5,random_state=0,criterion='gini',max_features='sqrt')
rfcl.fit(x_train,y_train)


# In[172]:


rfcl.score(x_train,y_train)


# In[173]:


rfcl.score(x_test,y_test)


# In[ ]:




