import streamlit as st
from data import *
from visuals_and_analysis import *
from prediction import future_prediction_lr, future_prediction_rf, future_prediction_nn
from prediction import *

def main():
    wages_dif = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ['Homepage', 'Introduction', 'Time Series', 'Box Plot',
                                         'Average Hourly Wages by Education Levels',
                                         'Hourly Wages by Gender',
                                         'Hourly Wages by Gender and Education Level',
                                         'Hourly Wages by Race and Education Level',
                                         'Future Prediction'])
    
    # Main Page Content
    if section == 'Homepage':
        display_dataset(wages_dif)
        display_statistics(wages_dif)
        display_time_series(wages_dif)
        display_box_plot(wages_dif)
        average_hourly_wages_by_education(wages_dif)
        hourly_wages_by_gender(wages_dif)
        hourly_wages_by_gender_and_education_level(wages_dif)
        hourly_wages_by_race_and_education_level(wages_dif)
    elif section == 'Introduction':
        display_dataset(wages_dif)
        display_statistics(wages_dif)
    elif section == 'Time Series':
        display_time_series(wages_dif)
    elif section == 'Box Plot':
        display_box_plot(wages_dif)
    elif section == 'Average Hourly Wages by Education Levels':
        average_hourly_wages_by_education(wages_dif)
    elif section == 'Hourly Wages by Gender':
        hourly_wages_by_gender(wages_dif)
    elif section == 'Hourly Wages by Gender and Education Level':
        hourly_wages_by_gender_and_education_level(wages_dif)
    elif section == 'Hourly Wages by Race and Education Level':
        hourly_wages_by_race_and_education_level(wages_dif)
    elif section == 'Future Prediction':
        st.markdown("<h2 style='color: orange;'>Wage Prediction for Next 50 Years (2023-2072)</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: lime;'>Select ML Model for Predictions:</h4>", unsafe_allow_html=True)
        model_selection = st.radio(
            "Select the model for prediction:",
            ('Linear Regression', 'Random Forest', 'Neural Network'),
            label_visibility='collapsed'
        )

        if model_selection == 'Linear Regression':
            future_prediction_lr(wages_dif, train_end_year=2022)
        elif model_selection == 'Random Forest':
            future_prediction_rf(wages_dif, train_end_year=2022)
        elif model_selection == 'Neural Network':
            future_prediction_nn(wages_dif, train_end_year=2022)


    # Footer
    st.write('---')
    st.write('Created with Streamlit')

if __name__ == '__main__':
    main()
