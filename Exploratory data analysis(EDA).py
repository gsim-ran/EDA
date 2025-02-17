#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv('titanic.csv')


# In[4]:


print("First few rows of the dataset:")
print(data.head())


# In[5]:


print("\nDataset Info:")
data.info()


# In[6]:


print("\nSummary Statistics:")
print(data.describe())


# In[7]:


print("\nMissing Values:")
print(data.isnull().sum())


# In[8]:


sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# In[9]:


sns.countplot(x='Survived', data=data, palette='pastel')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# In[10]:


sns.countplot(x='Survived', hue='Sex', data=data, palette='muted')
plt.title("Survival by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# In[11]:


sns.countplot(x='Survived', hue='Pclass', data=data, palette='cool')
plt.title("Survival by Passenger Class")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()


# In[12]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[13]:


sns.histplot(data['Age'].dropna(), kde=True, bins=30, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[14]:


sns.boxplot(x='Pclass', y='Fare', data=data, palette='Set3')
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()


# In[17]:


data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)


# In[18]:


print("\nMissing Values After Imputation:")
print(data.isnull().sum())


# In[ ]:




