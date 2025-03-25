import pandas as pd
import streamlit as st

# Set page title and favicon
st.set_page_config(page_title='Wages by Education in the USA', page_icon=':bar_chart:')

# Global variable for education levels
#education_levels = ['less_than_hs', 'high_school', 'some_college', 'bachelors_degree', 'advanced_degree']

@st.cache_data
def load_data():
    wages_dif = pd.read_csv('wages_by_education.csv').convert_dtypes()
    for col in wages_dif.columns:
        if wages_dif[col].dtype == 'int32':
            wages_dif[col] = wages_dif[col].astype('int64')
        if col != 'year':
            wages_dif[col] = wages_dif[col].astype(float)
        else:
            wages_dif[col] = wages_dif[col].astype(int)

    if wages_dif.isnull().sum().sum() > 0:
        wages_dif.dropna(inplace=True)
    return wages_dif

    
