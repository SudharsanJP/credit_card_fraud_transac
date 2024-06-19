import streamlit as st
import plotly.express as px

#)title
st.title(':orange[💮Handling imbalanced dataset in ML🌳]')
st.info("credit card fraud dataset")

#) reading dataset 1
import pandas as pd
df1 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\credit_card_fraud\fraudTest.csv")


#) to get the columns of dataframe 1
#df1.columns

#) to check the values of column in dataframe 1
#df1[['trans_date_trans_time','cc_num','merchant','category','amt','first','last','gender','street','city','state']]

#) to check the values of column in dataframe 1
#df1[['zip','lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time','merch_lat', 'merch_long', 'is_fraud']]

#) to check the fraud columns
#df1['is_fraud'].unique()

#) reading dataset 2
df2 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\credit_card_fraud\fraudTrain.csv")


#) dataframe 2 columns
#df2.columns

#) dataframe 1 columns
#df1.columns

#)concatenation of dataframes(1-2)
df = pd.concat([df1, df2], axis=0, ignore_index=True)

#) 21. checkbox with text
st.subheader("\n:green[1. Analysis dataset🌝]\n")
if (st.checkbox("original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")
  
    # Sample data
    data = df.head(5)

    # Display the table with highlighting
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))
    #) to get how much percentage of is_fraud = 0 data
    data_count1 = ((df['is_fraud'] == 0).sum()/len(df)) * 100
    st.code(f"class 0 = no fraud data {data_count1}\n")
    
    #) to get how much percentage of is_fraud = 1 data
    data_count2 = ((df['is_fraud'] == 1).sum()/len(df)) * 100
    st.code(f"class 1 = fraud data {data_count2}")

#) to get the data of 'trans_date_trans_time' columns
#df['trans_date_trans_time']

#) to get the year from 'trans_date_trans_time' column
list_tran = df['trans_date_trans_time'].tolist() #) converting item_date column into list
year_string = map(str, list_tran)
list_year_string = list(year_string)

index = 0
list_tran_year = []

while index < len(list_year_string):
    year1 = list_year_string[index]
    #) to pick up year
    trans_year = year1[0:4]
    list_tran_year.append(trans_year)
    index+=1
#list_tran_year

#) converting list into column
df['tran_year'] = list_tran_year


#) to check the type of data in each column
datatypes = df.dtypes

#)converting str column to int column 
df['tran_year'] = df['tran_year'].astype(str).astype(int)

#) to check the type of data in each column after type casting
#datatypes = df.dtypes

#) to get the hour from 'trans_date_trans_time' column
list_tran = df['trans_date_trans_time'].tolist() #) converting item_date column into list
tran_string = map(str, list_tran)
list_tran_string = list(tran_string)

index = 0
list_tran_hour = []

while index < len(list_tran_string):
    hour1 = list_year_string[index]
    #) to pick up year
    trans_hour = hour1[11:13]
    list_tran_hour.append(trans_hour)
    index+=1
#list_tran_hour

#) converting list into column
df['tran_hour'] = list_tran_hour

#)converting str column to int column 
df['tran_hour'] = df['tran_hour'].astype(str).astype(int)

#) to check the type of data in each column after type casting
datatypes = df.dtypes


#) unique value in tran_hour column
#df['tran_hour'].unique()

#) fraud column = 1 with transection hour more than 20
#df[(df['is_fraud'] == 1) & (df['tran_hour']>20)]

#) fraud column = 1 with transection hour less than 20
#df[(df['is_fraud'] == 1) & (df['tran_hour']<5)]

#) to check the null value
#df.isna().sum()

#) to get the city_pop column data
#df['city_pop']

#) scatter plot1
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#) converting df to list
list_city_pop =df['city_pop'].tolist()
list_amt = df['amt'].tolist()

#) converting list to np
np_city_pop = np.array(list_city_pop)
np_amt = np.array(list_amt)

if (st.checkbox("scatterplot")):
    #)showing original dataframe
    st.markdown("\n#### :blue[1.2 scatterplot]\n")
    fig,ax = plt.subplots(figsize=(15,8))
    ax.scatter(np_city_pop,np_amt,color = 'yellow')
    st.pyplot(fig)

#) to get max value in amt column
#df['amt'].max()

#)boxplot
if (st.checkbox("boxplot")):
    #) boxplot
    st.markdown("\n#### :red[1.3 boxplot]\n")
    fig = px.box(df,x = "tran_year",y="amt")
    st.plotly_chart(fig)

#) to get the amt value above 1300 at fraud ==1 
#df[(df['amt']>1300) & (df['is_fraud'] == 1)]

#) to get fraud ==1 at 2019
#df[(df['is_fraud'] == 1) & (df['tran_year']==2019)]

#) to get fraud ==1 at 2020
#df[(df['is_fraud'] == 1) & (df['tran_year']==2020)]

#) to get unique values in merchant column
#df['merchant'].unique()

#) to count 'fraud_Labadie LLC' in merchant column
#(df['merchant'] == 'fraud_Labadie LLC').sum()

#) to count 'fraud_Homenick LLC' in merchant column
#(df['merchant'] == 'fraud_Homenick LLC').sum()

#) dropping unncessary columns
df.drop(['Unnamed: 0','trans_date_trans_time','cc_num','first','last',
        'street','dob','trans_num','unix_time'],axis=1,inplace=True)

#) encoding the string column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols =['merchant','category','gender','city','state','job']
for col in cols:
    df[col] = le.fit_transform(df[col])
#df

#) classification ml models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


X = df.drop(['is_fraud'],axis=1)
y = df['is_fraud']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model =  DecisionTreeClassifier()
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

if (st.checkbox("Decision Tree Classifier")):
    #) DecisionTreeClassifier
    st.markdown("\n#### :red[1.4 DecisionTreeClassifier]\n")
    st.success('*******Train*******')
    st.write(f"Accuracy: {accuracy_score(y_train,train_pred)}")
    st.write(f"Precision: {precision_score(y_train,train_pred)}")
    st.write(f"Recall: {recall_score(y_train,train_pred)}")
    st.write(f"F1 Score: {f1_score(y_train,train_pred)}")
    
    st.success('*******Test*******')
    st.write(f"Accuracy: {accuracy_score(y_test,test_pred)}")
    st.write(f"Precision: {precision_score(y_test,test_pred)}")
    st.write(f"Recall: {recall_score(y_test,test_pred)}")
    st.write(f"F1 Score: {f1_score(y_test,test_pred)}")
    st.write("\n\n")


import numpy
from sklearn import metrics
import scikitplot as skplt
#) y train dataframe
#y_train

#) train prediction array
#train_pred

st.subheader(':orange[2. confusion matrix]📔')
selectBox=st.selectbox("evaluation: ", ['option',
                                    'for train',
                                    'for test'])

if selectBox == 'option':
    st.text("select next option")

elif selectBox == 'for train':
    #) confusion matrix for training
    st.markdown("\n#### :violet[2.1 training data]\n")
    cofusion_figure = plt.figure(figsize=(4,4))
    ax1 = cofusion_figure.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_train,train_pred,ax=ax1)
    st.pyplot(cofusion_figure, use_container_width=True)
#) y test dataframe
#y_test

#) test prediction array
#test_pred

elif selectBox == 'for test':
    #) confusion matrix for training
    st.markdown("\n#### :violet[2.2 testing data]\n")
    cofusion_figure = plt.figure(figsize=(4,4))
    ax1 = cofusion_figure.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test,test_pred,ax=ax1)
    st.pyplot(cofusion_figure, use_container_width=True)

#) to get the fraud dataframe
fraud_df = df[df['is_fraud'] == 1]
map_df = fraud_df[['lat','long']]
map_df.rename(columns={"lat": "latitude", "long": "longitude"}, inplace=True)
st.subheader("\n:blue[3. credit card fraud transaction geomap🚓:]\n")
if (st.button(':red[click here]')):
    columns = ['latitude','longitude']
    st.map(map_df)