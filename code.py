#!/usr/bin/env python
# coding: utf-8

# In[66]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[67]:


# read dataset
data_path = os.path.join(os.getcwd(), "data/healthcare-dataset-stroke-data.csv")
data = pd.read_csv(data_path)
data.head()


# In[68]:


data.info()


# In[69]:


# Data cleaning
data_cleaned = data.dropna(axis=0)
data_cleaned.tail()


# In[70]:


gender_count = data_cleaned.groupby("gender").size()
gender = data_cleaned[data_cleaned["stroke"] == 1].groupby("gender").size()

plt.figure(figsize=(15,10))
sns.barplot(gender.index, gender.values / gender_count.values[0:1])
plt.xlabel("Gender", size=20)
plt.ylabel("Stroke rate", size=20)
plt.show()


# In[71]:



plt.figure(figsize=(15,10))
plt.title("Age distribution in dataset", size=30)
sns.distplot(data_cleaned["age"], kde=True)
plt.xlabel("Age", size=15)
plt.show()


# In[72]:


stroked = data_cleaned[data_cleaned["stroke"] == 1]
plt.figure(figsize=(15,10))
plt.title("Stroke distribution by age", size=30)
sns.distplot(stroked["age"])
plt.xlabel("Age", size=15)
plt.show()


# In[73]:


hypertension = data_cleaned.groupby(["hypertension", "stroke"]).size().unstack()
plt.figure(figsize=(20,8))
plt.subplot(1, 2, 1)
for i, j in hypertension.iterrows():
    s = j.sum()
    plt.bar(i, j[1] / s)
plt.xticks(visible=False)
plt.legend(hypertension[0].index)
plt.xlabel("Hypertension", size=20)
plt.ylabel("Stroke rate", size=20)

heart_disease = data_cleaned.groupby(["heart_disease", "stroke"]).size().unstack()
plt.subplot(1, 2, 2)
for i, j in hypertension.iterrows():
    s = j.sum()
    plt.bar(i, j[1] / s)
plt.xticks(visible=False)
plt.legend(heart_disease[0].index)
plt.xlabel("Heart disease", size=20)
plt.show()


# In[74]:


hypertension = data_cleaned.groupby(["hypertension", "stroke"]).size().unstack()
hypertension


# In[75]:


work_type_counts = data_cleaned.groupby("work_type").size()
work_type = data_cleaned[data_cleaned["stroke"] == 1].groupby("work_type").size()
temp = work_type / work_type_counts
temp = temp.fillna(0)

plt.figure(figsize=(15,10))
plt.title("Stroke rate vs. work types", size=30)
sns.barplot(temp.index, temp.values)
plt.xlabel("Work type", size=20)
plt.ylabel("Stroke rate", size=20)
plt.xticks(size=15)
plt.show()


# In[76]:


residence_counts = data_cleaned.groupby("Residence_type").size()
residence_stroke_rate = data_cleaned[data_cleaned["stroke"] == 1].groupby("Residence_type").size() / residence_counts
residence_stroke_rate

plt.figure(figsize=(15,10))
plt.title("Stroke rate vs. residence types", size=30)
sns.barplot(residence_stroke_rate.index, residence_stroke_rate.values)
plt.xlabel("Residence type", size=20)
plt.ylabel("Stroke rate", size=20)
plt.xticks(size=15)
plt.show()


# In[77]:


data_cleaned["smoking_status"].unique()
smoker_counts = data_cleaned.groupby("smoking_status").size()
smoker_stroke_rate = data_cleaned[data_cleaned["stroke"] == 1].groupby("smoking_status").size() / smoker_counts
smoker_stroke_rate

plt.figure(figsize=(15,10))
plt.title("Stroke rate vs. smoke status", size=30)
sns.barplot(smoker_stroke_rate.index, smoker_stroke_rate.values)
plt.xlabel("Smoke status", size=20)
plt.ylabel("Stroke rate", size=20)
plt.xticks(size=15)
plt.show()


# In[78]:


residence_marriage = data_cleaned.groupby(["Residence_type", "ever_married"]).size().unstack()

plt.figure(figsize=(15,10))
sns.barplot(["Rural", "Urban"], [(residence_marriage.iloc[0]["Yes"] / sum(residence_marriage.iloc[0].values)), (residence_marriage.iloc[1]["Yes"] / sum(residence_marriage.iloc[1].values))])
plt.ylabel("Marriage rate", size=20)
plt.xlabel("Residence types", size=20)
plt.show()


# In[79]:


data_cleaned.head()


# In[80]:



plt.figure(figsize=(20,20))
plt.subplot(2, 1, 1)
sns.distplot(stroked["avg_glucose_level"])
plt.xlabel("Average glucose levels of stroke cases", size=20)
plt.subplot(2, 1, 2)
sns.distplot(data_cleaned["avg_glucose_level"])
plt.xlabel("Average glucose levels", size=20)
plt.show()


# In[81]:


stroke_bmi = pd.cut(data_cleaned[data_cleaned["stroke"] == 1]["bmi"], bins=[0, 18.5, 24, 28, 98]).value_counts()
bmi = pd.cut(data_cleaned["bmi"], bins=[0, 18.5, 24, 28, 98]).value_counts()

plt.figure(figsize=(15,10))
plt.title("Stroke rate vs. BMI", size=30)
sns.barplot(stroke_bmi.index, stroke_bmi.values / bmi.values)
plt.xlabel("BMI", size=20)
plt.ylabel("Stroke rate", size=20)
plt.xticks(size=15)
plt.show()


# In[82]:


data_cleaned.head()


# In[83]:


# Feature extraction

features = data_cleaned.drop(["id", "stroke"], axis=1)
target = data_cleaned["stroke"]

# Standardise
scaler = StandardScaler()
num_cols = features.select_dtypes(float)
features[num_cols.columns] = scaler.fit_transform(features[num_cols.columns])

# One hot encoding
features = pd.get_dummies(features)


# In[84]:


# Model training
rfc = RandomForestClassifier()
lr = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25)

rfc.fit(X_train, y_train)
lr.fit(X_train, y_train)

print("Random forest score: ", rfc.score(X_test, y_test))
print("Logistic regression score: ", lr.score(X_test, y_test))


# In[85]:


# Easter egg
married_count = data_cleaned.groupby("ever_married").size()
heart_disease = data_cleaned[data_cleaned["heart_disease"] == 1]
hyperstension = data_cleaned[data_cleaned["hypertension"] == 1]
heart_and_hyperstension = hyperstension[hyperstension["heart_disease"] == 1]

temp = hyperstension.groupby(["ever_married"]).size()

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
sns.barplot(married_count.index, temp / married_count.values)
plt.ylabel("Hypertension rate", size=15)
plt.xlabel("Ever married", size=15)
plt.xticks(size=10)

temp = heart_disease.groupby(["ever_married"]).size()
plt.subplot(2, 2, 2)
sns.barplot(married_count.index, temp / married_count.values)
plt.ylabel("Heart disease rate", size=15)
plt.xlabel("Ever married", size=15)
plt.xticks(size=10)

temp = heart_and_hyperstension.groupby(["ever_married"]).size()
plt.subplot(2, 2, 3)
sns.barplot(married_count.index, temp / married_count.values)
plt.ylabel("Heart disease & hypertension rate", size=15)
plt.xlabel("Ever married", size=15)
plt.xticks(size=10)

temp = data_cleaned[data_cleaned["stroke"] == 1].groupby("ever_married").size()
plt.subplot(2, 2, 4)
sns.barplot(married_count.index, temp / married_count.values)
plt.ylabel("Stroke rate", size=15)
plt.xlabel("Ever married", size=15)
plt.xticks(size=10)
plt.show()

