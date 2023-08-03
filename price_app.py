import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.model_selection import RandomizedSearchCV
import os
import joblib
import seaborn as sns
from matplotlib import rcParams

#Loading the cleaned data from the  excel file
price_data = pd.read_csv("https://raw.githubusercontent.com/nicholaswanipaul/streamlit/main/food_prices_cleaned.csv")

#Preparation of indepenent variables year, commodity and price flag which determine the final unit price
year_list = price_data['Year'].unique().tolist()
# increasing prediction year up to 2026,
x = len(year_list)
year_list.insert(x, 2024)
year_list.insert(x + 1, 2025)
year_list.insert(x + 2, 2026)

#price_rating = price_data['unit_price'].unique().tolist()
#category_list = price_data['category'].unique().tolist()
commodity_list = price_data['commodity'].unique().tolist()

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
priceflag_list = price_data['priceflag'].unique().tolist()

#Getting the min, average and max price from the cleaned dataset for visualization
min_unit_price=price_data['unit_price'].min()
avg_unit_price=price_data['unit_price'].mean()
max_unit_price=price_data['unit_price'].max()


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('South Sudan Food Prices Dashboard')
with st.sidebar:
    #this code will help the user select an item to predict from the list
    commodity=st.selectbox('Select commodity to Predict Price',commodity_list)
    year_option=st.selectbox('Select Prediction year', (year_list))
    price_flag_option=st.selectbox('Select Price Type', (priceflag_list))

    #This code will capture the selection from the user to be fed into the model for prediction
    if st.button('Click to Predict'):
        if commodity == "Beans red":
          x = 2
        elif commodity == "Cassava":
          x = 3
        elif commodity == "Cassava dry":
          x = 4
        elif commodity == "Charcoal":
          x = 5
        elif commodity == "Chicken":
          x = 6
        elif commodity == "Cowpeas":
          x = 7
        elif commodity == "Exchange rate":
          x = 8
        elif commodity == "Exchange rate unofficial":
          x = 9
        elif commodity == "Fish dry":
          x = 10
        elif commodity == "Fish fresh":
          x = 11
        elif commodity == "Fuel diesel":
          x = 12
        elif commodity == "Fuel diesel parallel market":
          x = 13
        elif commodity == "Fuel petrol-gasoline":
          x = 14
        elif commodity == "Fuel petrol-gasoline parallel market":
          x = 15
        elif commodity == "Groundnuts shelled":
          x = 16
        elif commodity == "Groundnuts unshelled":
          x = 17
        elif commodity == "Livestock cattle":
          x = 18
        elif commodity == "Livestock goat":
          x = 19
        elif commodity == "Livestock sheep":
          x = 20
        elif commodity == "Maize food aid":
          x = 21
        elif commodity == "Maize meal":
          x = 22
        elif commodity == "Maize white":
          x = 23
        elif commodity == "Meat beef":
          x = 24
        elif commodity == "Meat goat":
          x = 25
        elif commodity == "Milk fresh":
          x = 26
        elif commodity == "Millet white":
          x = 27
        elif commodity == "Milling cost Maize":
          x = 28
        elif commodity == "Milling cost sorghum":
          x = 29
        elif commodity == "Oil vegetable":
          x = 30
        elif commodity == "Okra dry":
          x = 31
        elif commodity == "Peas yellow":
          x = 32
        elif commodity == "Potatoes Irish":
          x = 33
        elif commodity == "Rice":
          x = 34
        elif commodity == "Salt":
          x = 35
        elif commodity == "Sesame":
          x = 36
        elif commodity == "Sorghum":
          x = 37
        elif commodity == "Sorghum brown":
          x = 38
        elif commodity == "Sorghum flour":
          x = 39
        elif commodity == "Sorghum food aid":
          x = 40
        elif commodity == "Sorghum red":
          x = 41
        elif commodity == "Sorghum white":
          x = 42
        elif commodity == "Sugar brown":
          x = 43
        elif commodity == "Sugar food aid":
          x = 44
        elif commodity == "Wage":
          x = 45
        elif commodity == "Wheat flour":
          x = 46
        else:
          x = 0  
        #st.write(x)
        if price_flag_option == "actual":
          y = 1
        elif price_flag_option== "aggregate":
          y = 3
        else:
          y = 2

        # After the user's choices are captured, a new pandas dataframe data structure is created to capture the users choices
        data = dict(Year=year_option,priceflag=y,commodity=x)
        df = pd.DataFrame(data, index=[0])
        #df.info()
        #Here the model is loaded and the user choices are fed into the model for prediction
        loaded_model=joblib.load("foodprices_model.joblib")
        predicted_price=loaded_model.predict(df)
         #The prediction value is shown to the user
        st.write(predicted_price)
    else:
        st.write("Predicted Price will be diaplayed here")

# Row A
st.markdown('### Main Metrics for Unit Food Prices in South Sudan')
col1, col2, col3 = st.columns(3)
col1.metric("Minimum Price recorded", round(min_unit_price,2), "SSP")
col2.metric("Average Price Recorded",round(avg_unit_price, 2), "SSP")
col3.metric("Maximum Unit Price Recorded", round(max_unit_price, 2), "SSP")
stocks=200
# Row B

c1, c2 = st.columns((3,3))
with c1:
    #This code will display geospatial information on the dashboard
    st.markdown('### Locations of Main Markets in South Sudan')
    mapbox_access_token = 'pk.eyJ1IjoiMDEyMzQ1Njc4OTA5ODc2NTQzMjEwIiwiYSI6ImNqdmdlZnZyMzAzcTQ0OHBjOGN0ZTl1ZW4ifQ.AHB-GJ3EtYeUrHIvtGBDkg'
    px.set_mapbox_access_token(mapbox_access_token)
    fig = px.scatter_mapbox(price_data,mapbox_style='satellite',lat="latitude", lon="longitude", hover_name="admin2", hover_data=["admin1", "admin2"], color_discrete_sequence=["fuchsia"], zoom=5, height=400, text='admin2')
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
with c2:
    #Bar Graph for Average food Prices per category
    st.markdown('### Average Price for per Food Category')
    st.pyplot(price_data.groupby("category")["price"].mean().plot(kind='bar', x='price', y="category",fontsize=3, figsize = (3,2)).figure, use_container_width=True)


# Row C
coln1, coln2 = st.columns((0.5,0.5))
with coln1:
    #visualizations
     st.markdown('### Average Price per State')
     st.pyplot(price_data.groupby("admin1")["price"].mean().plot(kind='bar').figure, )
with coln2:
     st.pyplot(price_data.groupby("market")["price"].mean().plot(kind='bar', fontsize=3).figure, )
