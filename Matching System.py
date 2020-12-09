#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
dataset=pd.read_csv("D:\PyCharm Community Edition 2020.1.3\datasets\Speed Dating Data.csv",encoding="unicode_escape")

pd.set_option('display.max_columns', None)
dataset.head()


# In[2]:


#drop the repeated waves
data1=dataset.drop_duplicates(subset=["iid"])


# In[3]:


background=data1[data1[["field_cd","mn_sat","income"]].notnull()]
background=background[["field_cd","mn_sat","income"]]
background=background.dropna()
for c in range(len(background["income"])):
    numb=re.findall(r'[0-9]+',background.iloc[c,2])
    income="0"
    for num in numb[0:-1]:
        income=income+num
    background.iloc[c,2]=int(income)
for c in range(len(background["mn_sat"])):
    numb=re.findall(r'[0-9]+',background.iloc[c,1])
    income="0"
    for num in numb[0:-1]:
        income=income+num
    background.iloc[c,1]=int(income)


# In[4]:


ssd={}
for k in range(1,15):
    model=KMeans(n_clusters=k,max_iter=1000).fit(background)
    ssd[k]=model.inertia_
plt.figure()
plt.plot(list(ssd.keys()), list(ssd.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[5]:


model=KMeans(n_clusters=5,max_iter=1000).fit(background)
background["cluster_intel"]=model.labels_


# In[6]:


data2=pd.concat([data1,background],axis=1, ignore_index=False)
data2


# In[7]:


interest=data1[['sports','tvsports','exercise','dining','museums','art','hiking','gaming','clubbing','reading','tv','theater',
               'movies','concerts','music','shopping','yoga']]
interest=interest.dropna()


# In[8]:


ssd={}
for k in range(1,20):
    model=KMeans(n_clusters=k,max_iter=1000).fit(interest)
    ssd[k]=model.inertia_
plt.figure()
plt.plot(list(ssd.keys()), list(ssd.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[9]:


model=KMeans(n_clusters=10,max_iter=1000).fit(interest)
interest["cluster_inte"]=model.labels_


# In[58]:


data3=pd.concat([data2,interest],axis=1, ignore_index=False)
data3


# In[61]:


data3=data3.dropna(subset=["cluster_inte","cluster_intel"])
data3=data3.set_index(data3["iid"])


# In[54]:


def matching_system(datasource,career,place,iid,aptitude):
    career_map={"Lawyer":"1" , 
"Academic/Research":"2", 
"Psychologist":"3",
"Doctor/Medicine": "4", 
"Engineer":"5",
"Creative Arts/Entertainment":"6", 
 "Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin":"7",
 "Real Estate":"8",
 "International/Humanitarian Affairs":"9", 
"Undecided":"10", 
"Social Work":"11",
"Speech Pathology":"12",
"Politics":"13",
"Pro sports/Athletics":"14",
"Other":"15",
"Journalism":"16",
"Architecture":"17"}
    career_c=career_map[career]
    filtered_1=datasource[datasource["career_c"]==career_c]
    if aptitude=="straight":
        filtered_1=filtered_1[filtered_1["gender"]!=datasource.loc[iid,"gender"]]
    elif aptitude=="gay":
        filtered_1=filtered_1[filtered_1["gender"]==datasource.loc[iid,"gender"]]
    elif aptitude=="biual":
        filtered_1=filtered_1
    filtered_2=filtered_1[filtered_1["from"]==place]
    list_intel=filtered_2[filtered_2["cluster_intel"]==datasource.loc[iid,"cluster_intel"]]["iid"].tolist()
    list_inte=filtered_2[filtered_2["cluster_inte"]==datasource.loc[iid,"cluster_inte"]]["iid"].tolist()
    share=[]
    for i in list_intel:
        if i in list_inte:
            share.append(i)
        else:
            continue
    if len(share)==0:
        if datasource.loc[iid,"intel1_1"]>datasource.loc[iid,"sha1_1"]:
            share.append(list_intel)
        else:
            share.append(list_inte)
    return share


# In[51]:


share1=matching_system(data3,"Lawyer","Chicago",iid=361,aptitude="straight")
share1


# In[48]:


iid=361
career_map={"Lawyer":1, 
"Academic/Research":2, 
"Psychologist":3,
"Doctor/Medicine": 4, 
"Engineer":"5",
"Creative Arts/Entertainment":"6", 
 "Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin":"7",
 "Real Estate":"8",
 "International/Humanitarian Affairs":"9", 
"Undecided":"10", 
"Social Work":"11",
"Speech Pathology":"12",
"Politics":"13",
"Pro sports/Athletics":"14",
"Other":"15",
"Journalism":"16",
"Architecture":"17"}
#career_c=career_map[career]
#filtered_2=data3[data3["career_c"]==career_c]
filtered_2=data3[data3["gender"]!=data3.loc[iid,"gender"]]
#filtered_2.set_index(filtered_2["iid"])
#filtered_2=filtered_1[filtered_1["from"]==place]
list_intel=filtered_2[filtered_2["cluster_intel"]==data3.loc[iid,"cluster_intel"]]["iid"].tolist()
list_inte=filtered_2[filtered_2["cluster_inte"]==data3.loc[iid,"cluster_inte"]]["iid"].tolist()
share=[]
for i in list_intel:
    if i in list_inte:
        share.append(i)
    else:
        continue
if len(share)==0:
    if data3.loc[iid,"intel1_1"]>data3.loc[iid,"shar1_1"]:
        share.append(list_intel)
    else:
        share.append(list_inte)

