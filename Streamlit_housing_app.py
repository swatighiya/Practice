# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
st.title('DreamHomes.com')
st.text('Find your dream homes here')
df = pd.read_csv('kc_house_data.csv')
st.image('House.jpg')
st.video('https://www.youtube.com/watch?v=zvL0UmvPSEA')
price_set = st.slider("Price Range", min_value = int(df['price'].min()), max_value = int(df['price'].max()), step = 1, value = int(df['price'].min()))
st.text ("Price selected is "+str(price_set))
fig = px.scatter_mapbox(df.loc[df['price']<price_set],lat='lat',lon='long',color = 'sqft_living', size = 'price')
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig)
st.header("Price Predictor")
sel_box_var = st.selectbox("Select Method", ['Linear','Ridge','Lasso'],index=0)
multi_var= st.multiselect("Select additional variables for accuracy", ['sqft_living','sqft_lot','sqft_basement'])
df_new = []
df_new = df[multi_var]
df_new['bedrooms']=df['bedrooms']
df_new['bathrooms']=df['bathrooms']
if sel_box_var == 'Linear':
    x=df_new
    y=df['price']
    model=LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(x_train,y_train)
    y_pred=reg.predict(x_test)
    st.text("Intercept = "+str(reg.intercept_))
    st.text("Coefficient = "+str(reg.coef_))
    st.text("R^2 = "+str(r2_score(y_test,y_pred)))
elif sel_box_var == 'Ridge':
    x=df_new
    y=df['price']
    model=Ridge()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(x_train,y_train)
    y_pred=reg.predict(x_test)
    st.text("Intercept = "+str(reg.intercept_))
    st.text("Coefficient = "+str(reg.coef_))
    st.text("R^2 = "+str(r2_score(y_test,y_pred)))
else: 
    x=df_new
    y=df['price']
    model=Ridge()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(x_train,y_train)
    y_pred=reg.predict(x_test)
    st.text("Intercept = "+str(reg.intercept_))
    st.text("Coefficient = "+str(reg.coef_))
    st.text("R^2 = "+str(r2_score(y_test,y_pred)))
st.set_option('deprecation.showPyplotGlobalUse',False)
sns.regplot(y_test,y_pred)
st.pyplot()
count = 0
pred_value=0
for i in df_new.keys():
    try:
        val=st.text_input("Enter number of "+i)
        pred_value=pred_value+float(val)*reg.coef_[count]
        count=count+1
    except:
        pass
st.text('Predicted prices are: '+str(pred_value+reg.intercept_))
st.header("Application Details")
img = st.file_uploader("Upload application")
st.text("Details for the representative to contact you")
adress=st.text_area("Please enter your adress here")
date=st.date_input("Please enter the date")
time=st.time_input("Please enter the time")
if st.checkbox("I comfirm the date and time",value=False):
    st.write("Thanks for confirming")
st.number_input("Please rate our site",min_value=1,max_value=10,step=1)





    
    
    



