# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:35:00 2020

@author: YBlachonpro
"""
import pandas as pd
import sklearn as sk
from sklearn import linear_model , tree , metrics, preprocessing
import matplotlib.pyplot as plt

### methods 

def convert_into_float(string):
    if type(string)==float or type(string)==int:
        return string
    return float(string.replace(',','.'))
### data prep 

df=pd.read_csv("challenge_youtube_toxic.csv",sep=";",encoding='latin1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df=df.loc[~(df["nbrMotInsulte"]>=2500 )]



    
# Ajout de nouvelles variables explicatives
 # moyenne nbr insulte par channel  
channels=pd.DataFrame()
channels["nbrMotInsulte_channel"]=df.groupby("channel_id").mean()["nbrMotInsulte"]
df=df.join(channels,on="channel_id")
 # moyenne nbr insulte par categorie / largeur d'audience
category_inst=pd.DataFrame()
category_new=pd.DataFrame()
category_inst["nbrMotInsulte_taille_categ"]=df.groupby("categ_inst").mean()["nbrMotInsulte"]
category_new["nbrMotInsulte_type_categ"]=df.groupby("categorie_new").mean()["nbrMotInsulte"]
df=df.join(category_inst,on="categ_inst")
df=df.join(category_new, on="categorie_new")

output=list(df["nbrMotInsulte"])
df.drop(["nbrMotInsulte","categ_inst","categorie_new","channel_name","channel_id","video_id_court","video_id"],axis=1,inplace=True)
df=df.applymap(convert_into_float)               
training=df[:int(df.shape[0]//1.42)]
test=df[int(df.shape[0]//1.42):]



scaler = preprocessing.StandardScaler()
# Fit on training set only.
scaler.fit(training)
# Apply transform to both the training set and the test set.
training = scaler.transform(training)
test = scaler.transform(test)


    


out_train=output[:int(df.shape[0]//1.42)]
#cv=df[27600:27600+9200]
#out_cv=output[27600:27600+9200]
out_test=output[int(df.shape[0]//1.42):]


model=sk.linear_model.LinearRegression().fit(training,out_train)
#model2=sk.tree.DecisionTreeRegressor().fit(training,out_train)

print(model.score(test,out_test))
#print(model2.score(test,out_test))
predicted=[]
a=model.predict(test)
for i in range(len(test)):
    predicted.append(a[i])


#Prise en compte de la chaine : 


    
