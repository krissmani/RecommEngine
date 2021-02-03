# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:16:23 2020

@author: MY394222
"""



import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import OneHotEncoder
from numpy import dot
from numpy.linalg import norm
from pprint import pprint 
from statistics import mean
import json



def create_data(filename,sheet,new_record):
    df_product = pd.read_excel(filename, sheet_name=sheet)
    rescale=MinMaxScaler(feature_range=(1, 5))
    df_product['Price ( Per Month/User)'] = rescale.fit_transform(df_product['Price ( Per Month/User)'].values.reshape(-1, 1))  
    df=pd.read_excel(filename,encoding="utf-8").drop_duplicates()
    df = df.append(new_record, ignore_index = True)  
    dataset=df.iloc[:, :-2]
    return dataset,df,df_product

def min_max_scale(dataset):
    scaler=MinMaxScaler()
    scaled = scaler.fit_transform(dataset['Size of contact center'].values.reshape(-1, 1))
    return scaled

def onehot_encoding(x_train,features,cleanup):

    x_train_label = x_train.replace(cleanup)

    enc = OneHotEncoder(handle_unknown='ignore')
    data=enc.fit_transform(x_train[features]).toarray()
    dfOneHot = pd.DataFrame(data) 
    scaled=min_max_scale(x_train)
    x_train = pd.concat([pd.DataFrame(scaled), dfOneHot], axis=1)
    return x_train,x_train_label,enc

def get_similarities(dataset,df,x_train_ohe,item):
    cosine_similarities= linear_kernel(x_train_ohe.values,Y=None)
    # cosine_similarities= cosine_similarity(x_train_ohe.values,Y=None)
    results = {}
    for idx, row in dataset.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], dataset['Cust name'][i],df['Recommended platform'][i]) for i in similar_indices]

        results[row['Cust name']] = similar_items[1:]
    return results[item]

def item(id,x_train_label):
    record=list(x_train_label.loc[x_train_label['Cust name'] == id].values[0])[3:]
    return record
def get_cust_prod_score(result):
    pro_results={}
    for score,cust,prod in result:
        if prod not in pro_results:
             pro_results[prod]=[(cust,score)]
        else:
            pro_results[prod].append((cust,score))
            
    return pro_results
        

def get_recommendation(fields,df_product,item_id,pro,x_train_label):
    
    final_recom={}    
    for plat in df_product["Product"].values:
        product_data=list(df_product[df_product["Product"] == plat][fields.values()].values[0])
        # final_recom[plat]=round(dot(item(item_id), product_data),2)
        final_recom[plat]=round(dot(item(item_id,x_train_label), product_data)/(norm(item(item_id,x_train_label))*norm(product_data)),2)
    
    sort_recom= [{'Recommended platform':y,'Product similarity score':z*100} for y,z in sorted(final_recom.items(), key=lambda x: x[1], reverse=True)[0:3]]
    final_rec=[]
    for i in sort_recom:
        try:
            i["Highest maching customer"]=pro[i['Recommended platform']][0][0]
            i["Max customer-Product similarity score"]=str(round((pro[i['Recommended platform']][0][1]/13.224377)*100,2)) 
            i["Avg customer-Product similarity score"]=round(((sum(score for cust,score in pro[i['Recommended platform']])/len(pro[i['Recommended platform']]))/13.224377)*100,2)
            
        except Exception as e:
            print(e)
            pass
        final_rec.append(i)
    return final_rec

def get_results(new_record):
    try:
        filename="CC Prod reco Training Data v3.xlsx"
        sheet="Product data"
        feature_file="feature_mapping.json"
        if type(new_record)==dict and len(new_record.keys())== 16:
            item_id=new_record["Cust name"]
            dataset,df,df_product=create_data(filename,sheet,new_record)
            with open(feature_file,"r") as f:
                data=json.loads(f.read())
            x_train_ohe,x_train_label,ohe_encoder=onehot_encoding(dataset,data["categorical_features"],data["cust_requirement_scaling"])
            result=get_similarities(dataset,df,x_train_ohe,item_id)
            pro=get_cust_prod_score(result)
            output=get_recommendation(data["cust_prod_mapping"],df_product,item_id,pro,x_train_label)
            return {"recommendations":output,"status":"success"}
        else:
            return {"recommendations":[],"status":"failed","error":"Incorrect input"}
    except Exception as e:  
        return {"recommendations":[],"status":"failed","error":e}
                   

if __name__ =="__main__":
    try:
        filename="CC Prod reco Training Data v3.xlsx"
        sheet="Product data"
        feature_file="feature_mapping.json"
        num=5
        
        print("\n\n------------------ Welcome to CC Recommendation Engine V1.0 2020 ------------------")
        
       
        for i in range(0,5):
            new_record=input("\n\nPlease enter the customer requirement in json format with key-value pairs : ")
            try:
                if type(new_record)==str and len(eval(new_record).keys())== 16:
                    item_id=eval(new_record)["Cust name"]
                    new_record=eval(new_record)
                    break
                else:
                    print("Please enter valid input; keys or format is incorrect.")
            except :
                pass
           
        dataset,df,df_product=create_data(filename,sheet,new_record)
        with open(feature_file,"r") as f:
            data=json.loads(f.read())
    except Exception as e:
        print("filename's may be incorrect. Actual error is ",e )
        
    try:
        x_train_ohe,x_train_label,ohe_encoder=onehot_encoding(dataset,data["categorical_features"],data["cust_requirement_scaling"])
        result=get_similarities(dataset,df,x_train_ohe,item_id)
        pro=get_cust_prod_score(result)
        output=get_recommendation(data["cust_prod_mapping"],df_product,item_id,pro)
        print("\n\n","Please find the below recommendations for customer : ",item_id,"\n")
        # print("---------------------------------------------------------------------")
        pprint(output)
        # print("---------------------------------------------------------------------")
    except Exception as e:
        print("Error in processing,Actual error is ",e)
    
    