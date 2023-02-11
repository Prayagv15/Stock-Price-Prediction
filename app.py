# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:58:07 2023

@author: Prayag V K
"""

import streamlit as st
import pandas as pd
pd.options.display.max_rows=10
import datetime
from PIL import Image
import pickle
import statsmodels
import plotly.express as px
import plotly.graph_objects as go

# Dataset
df=pd.read_csv("AAPL.csv")
df["Date"]=pd.to_datetime(df["Date"])
df["Date"]=df["Date"].dt.date
df.set_index("Date",inplace=True)
df.index.names=["Date"]
data=df
data1=df.drop(['Open','High','Low','Close','Volume'],axis=1)

# Model
loaded_model=pickle.load(open('modelf.pkl','rb'))

# Main Page.
st.title("**_Stock Price Prediction_** 📈")
st.markdown("![Alt Text](https://media.giphy.com/media/BwiHzMejx0420yUWgh/giphy-downsized-large.gif)")

inc=Image.open("OIP.jpg")

with st.expander("About"):
    st.image(inc,width=150)
    st.subheader("About Dataset")
    st.markdown('<div style="text-align:justify;">The dateset used for predicting the stock price is of AAPLE.Inc which is the data for the period of almost eight years from March-2012 to December-2019.</div>',unsafe_allow_html=True)
    st.write('')
    st.markdown('<div style="text-align:justify;">For time series analysis we need to have date as index. It has been observed that there are 252 days each year when stocks are traded.</div>',unsafe_allow_html=True)
    tab1,tab2=st.tabs(['Dataset','Plots'])
    with tab1:
        st.dataframe(data)
    with tab2:
        st.caption("Plots of *_Close_, _Open_, _High_, _Low_.*")
        df1=px.data.gapminder()
        fig=px.line(data,x=data.index,y=["Close","Open","High","Low"])
        st.plotly_chart(fig, theme=None,use_container_width=True)
with st.expander("Stock Price Prediction"):
    st.subheader("Time series analysis in Stock Price Prediction")
    st.markdown('<div style="text-align: justify;">Stock price prediction is an important topic in finance and economics. Stock price prediction is regarded as one of most difficult task to accomplish in financial forecasting due to the complex natrue of stock market.\n</div>',unsafe_allow_html=True)
    st.write(' ')
    st.markdown('<div style="text-align:justify;">Stock price analysis has been a critical area of research and is one of the top applications of machine learning. Stock price prediction using machine learning helps you discover the future value of the company and other financial assets traded on an exchange.</div>',unsafe_allow_html=True)
    st.write(' ')
    st.markdown('<div style="text-align: justify;">Time series forecasting is a method used to predict future valuess based on past data and trends.</div>',unsafe_allow_html=True)
    
# Side bar

data=st.sidebar.selectbox("Select the type of Data",('Original','Predicted'))
st.sidebar.caption('You are viewing Adjusted Close Price')
select=st.sidebar.radio('Select',('Tabular','Graphical'))

if data=="Original":
    if st.sidebar.button("Show"):
        st.subheader("Adjusted Close Prices")
        if select=="Tabular":
            data1.columns.name=data1.index.name
            data1.index.name=None
            st.write(data1.to_html(),unsafe_allow_html=True)
        else:
            df1=px.data.gapminder()
            fig2=px.line(data1,x=data1.index,y=["Adj Close"])
            st.plotly_chart(fig2, theme=None,use_container_width=True)
else:
    days=st.sidebar.slider('Select days for forecasting',min_value=1,max_value=30)
    predicted=loaded_model.forecast(days)
    indices=list(predicted.index)
    pred=pd.DataFrame(predicted,index=pd.Index(indices,name="Date"))
    pred.index=pd.date_range('2019-12-31',periods=days)
    pred.columns.name=pred.index.name
    pred.index.name=None
    pred.rename(columns={'predicted_mean':'Adj Close'},inplace=True) #Renaming the predicted Dataframe
    if st.sidebar.button('Predict'):
        st.subheader("Results: _Forecasted Prices_")
        if select=="Tabular":
            col1,col2=st.columns([2,3])
            col1.subheader(" ")
            col1.write(pred.to_html(),unsafe_allow_html=True)
            col2.subheader(" ")
            df1=px.data.gapminder()
            fig3=px.line(pred,x=pred.index,y=["Adj Close"])
            col2.plotly_chart(fig3,theme=None,use_container_width=True)
        else:
            df1=px.data.gapminder()
            fig4=px.line(data1,x=data1.index,y=data1["Adj Close"],color_discrete_sequence=["blue"],labels="original")
            fig5=px.line(pred,x=pd.date_range('2019-12-31',periods=days),y=pred['Adj Close'],color_discrete_sequence=["red"],labels="predicted")
            fig6=px.line(data1[-100:],x=data1[-100:].index,y=data1["Adj Close"][-100:],color_discrete_sequence=["blue"],labels="original")
            
            fig=go.Figure(data=fig6.data+fig5.data)
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(
                title="AAPL.Inc",
                xaxis_title="Date",
                yaxis_title="Adj Close",
                legend_title="Legend",
                height=700
                )
            fig7=go.Figure(data=fig4.data+fig.data)
            fig7.update_xaxes(rangeslider_visible=True)
            fig7.update_layout(
                title="AAPL.Inc",
                xaxis_title="Date",
                yaxis_title="Adj Close",
                legend_title="Legend",
                height=700
                )
            tab1,tab2=st.tabs(['Forecasted Prices with past 100 days',"Forecasted Prices with entire data"])
            with tab1:
                st.plotly_chart(fig,theme=None,use_container_width=True)
            with tab2:
                st.plotly_chart(fig7,theme=None,use_container_width=False)
    else:
        pass