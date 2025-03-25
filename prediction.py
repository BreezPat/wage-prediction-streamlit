import streamlit as st
import pandas as pd
import numpy as np
from data import *
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV


def future_prediction_lr(wage_dif, train_end_year=2022):
    """Predicts wages for the next 50 years based on the selected education level."""
    
    
    # Select the columns we're interested in
    columns = ['year', 'men_high_school', 'women_high_school',
               'men_bachelors_degree', 'women_bachelors_degree',
               'men_advanced_degree', 'women_advanced_degree']
    data = wage_dif[columns]

    # Normalize the data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data[['year']])
    scaled_y = scaler_y.fit_transform(data.drop('year', axis=1))

    # Split the data into features and targets
    X = scaled_x  # Year
    y = scaled_y  # Wages

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Linear Regression model
    lr_model = LinearRegression()

    # Train the Linear Regression model
    lr_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred_test = lr_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Test R-squared', 'Test RMSE'],
        'Value': [test_r2, test_rmse]
    })


    st.table(metrics_df)


    #st.write(f"Test R-squared: {test_r2}")
    #st.write(f"Test RMSE: {test_rmse}")

    # Create a DataFrame for the next 50 years and normalize the years
    future_years = pd.DataFrame({'year': range(train_end_year + 1, train_end_year + 51)})
    scaled_future_years = scaler_x.transform(future_years)

    # Predict future wages using the trained Linear Regression model
    future_wages_predictions_lr = lr_model.predict(scaled_future_years)

    # Inverse transform the predictions to get the actual wage values
    future_wages_lr = pd.DataFrame(scaler_y.inverse_transform(future_wages_predictions_lr), columns=columns[1:])
    future_wages_lr['year'] = future_years['year']

    # Radio button for selecting education level
    st.markdown("<h4 style='color: lime;'>Select Education Level for Prediction:</h4>", unsafe_allow_html=True)
    education_level_prediction = st.radio(
        "Select Education Level for Prediction:",
        ('High School', 'Bachelor\'s Degree', 'Advanced Degree'),
        label_visibility='collapsed'
    )

    # Mapping radio button selection to column names
    education_level_columns = {
        'High School': ['men_high_school', 'women_high_school'],
        'Bachelor\'s Degree': ['men_bachelors_degree', 'women_bachelors_degree'],
        'Advanced Degree': ['men_advanced_degree', 'women_advanced_degree']
    }

    selected_columns = education_level_columns[education_level_prediction]

    # Plotting the line chart for the selected education level using Plotly
    fig = px.line(future_wages_lr, x='year', y=selected_columns, labels={'value': 'Hourly Wage', 'variable': 'Education Level'},
                  title=f'Future Wage Predictions - {education_level_prediction}',
                  color_discrete_sequence=['blue', 'pink'])
    fig.update_layout(
        legend_title_text='Education Level',
        width=1000,  # Set the width of the chart
        height=600   # Set the height of the chart
    )

    fig.update_traces(line=dict(width=4))  # Increase the line thickness
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Hourly Wage')


    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print out the table of predicted values
    st.dataframe(future_wages_lr[['year'] + selected_columns])

    return future_wages_lr





def future_prediction_rf(wages_dif, train_end_year=2022):
    """Predicts wages for the next 50 years using a Random Forest."""
    # Select the columns we're interested in
    columns = ['year', 'men_high_school', 'women_high_school',
               'men_bachelors_degree', 'women_bachelors_degree',
               'men_advanced_degree', 'women_advanced_degree']
    data = wages_dif[columns]

    # Normalize the data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data[['year']])
    scaled_y = scaler_y.fit_transform(data.drop('year', axis=1))

    # Split the data into features and targets
    X = scaled_x  # Year
    y = scaled_y  # Wages

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred_rf = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Test R-squared', 'Test RMSE'],
        'Value': [rf_r2, rf_rmse]
    })

    st.table(metrics_df)

    # Create a DataFrame for the next 50 years and normalize the years
    future_years = pd.DataFrame({'year': range(train_end_year + 1, train_end_year + 51)})
    scaled_future_years = scaler_x.transform(future_years)

    # Predict future wages using the trained Random Forest model
    future_wages_predictions_rf = rf_model.predict(scaled_future_years)

    # Inverse transform the predictions to get the actual wage values
    future_wages_rf = pd.DataFrame(scaler_y.inverse_transform(future_wages_predictions_rf), columns=columns[1:])
    future_wages_rf['year'] = future_years['year']

    # Radio button for selecting education level
    st.markdown("<h4 style='color: lime;'>Select Education Level for Prediction:</h4>", unsafe_allow_html=True)
    education_level_prediction = st.radio(
        "Select Education Level for Prediction:",
        ('High School', 'Bachelor\'s Degree', 'Advanced Degree'),
        label_visibility='collapsed'
    )

    # Mapping radio button selection to column names
    education_level_columns = {
        'High School': ['men_high_school', 'women_high_school'],
        'Bachelor\'s Degree': ['men_bachelors_degree', 'women_bachelors_degree'],
        'Advanced Degree': ['men_advanced_degree', 'women_advanced_degree']
    }

    selected_columns = education_level_columns[education_level_prediction]

    # Plotting the line chart for the selected education level using Plotly
    fig = px.line(future_wages_rf, x='year', y=selected_columns, labels={'value': 'Hourly Wage', 'variable': 'Education Level'},
                  title=f'Random Forest Predictions - {education_level_prediction}',
                  color_discrete_sequence=['blue', 'pink'])
    fig.update_layout(
        legend_title_text='Education Level',
        width=1000,  # Set the width of the chart
        height=600   # Set the height of the chart
    )
    fig.update_traces(line=dict(width=4))  # Increase the line thickness
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Hourly Wage')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Print out the table of predicted values
    st.dataframe(future_wages_rf[['year'] + selected_columns])

    return future_wages_rf



def future_prediction_nn(wages_dif, train_end_year=2022):
    """Predicts wages for the next 50 years using a Neural Network."""
    # Select the columns we're interested in
    columns = ['year', 'men_high_school', 'women_high_school',
               'men_bachelors_degree', 'women_bachelors_degree',
               'men_advanced_degree', 'women_advanced_degree']
    data = wages_dif[columns]

    # Normalize the data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data[['year']])
    scaled_y = scaler_y.fit_transform(data.drop('year', axis=1))

    # Split the data into features and targets
    X = scaled_x  # Year
    y = scaled_y  # Wages

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the MLPRegressor model
    mlp_model = MLPRegressor(random_state=42)

    # Set up a grid search for hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(30, 30)],
        'activation': ['relu'],
        'solver': ['adam'],
        'max_iter': [1000],
        'learning_rate_init': [0.01]
    }
    grid_search = GridSearchCV(mlp_model, param_grid, cv=3, scoring='neg_mean_squared_error')

    # Train the model using grid search
    grid_search.fit(X_train, y_train)

    # Predict on the testing set using the best MLPRegressor model
    y_pred_mlp = grid_search.predict(X_test)

    # Calculate the evaluation metrics for MLPRegressor
    mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    mlp_r2 = r2_score(y_test, y_pred_mlp)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Test R-squared', 'Test RMSE'],
        'Value': [mlp_r2, mlp_rmse]
    })

    st.table(metrics_df)

    # Create a DataFrame for the next 50 years and normalize the years
    future_years = pd.DataFrame({'year': range(train_end_year + 1, train_end_year + 51)})
    scaled_future_years = scaler_x.transform(future_years)

    # Predict future wages using the trained Neural Network model
    future_wages_predictions_nn = grid_search.predict(scaled_future_years)

    # Inverse transform the predictions to get the actual wage values
    future_wages_nn = pd.DataFrame(scaler_y.inverse_transform(future_wages_predictions_nn), columns=columns[1:])
    future_wages_nn['year'] = future_years['year']


    # Radio button for selecting education level
    education_level_prediction = st.radio(
        "Select Education Level for Prediction:",
        ('High School', 'Bachelor\'s Degree', 'Advanced Degree'),
        label_visibility='collapsed'
    )

    # Mapping radio button selection to column names
    education_level_columns = {
        'High School': ['men_high_school', 'women_high_school'],
        'Bachelor\'s Degree': ['men_bachelors_degree', 'women_bachelors_degree'],
        'Advanced Degree': ['men_advanced_degree', 'women_advanced_degree']
    }

    selected_columns = education_level_columns[education_level_prediction]

    # Plotting the line chart for the selected education level using Plotly
    fig = px.line(future_wages_nn, x='year', y=selected_columns, labels={'value': 'Hourly Wage', 'variable': 'Education Level'},
                  title=f'Neural Network Predictions - {education_level_prediction}',
                  color_discrete_sequence=['blue', 'pink'])
    fig.update_layout(
        legend_title_text='Education Level',
        width=1000,  # Set the width of the chart
        height=600   # Set the height of the chart
    )

    fig.update_traces(line=dict(width=4))  # Increase the line thickness
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Hourly Wage')

    # Display the plot in the Streamlit app
    st.plotly_chart(fig)

    # Display the predicted future wages for MLPRegressor
    st.write("\nPredicted future wages for MLPRegressor:")
    st.dataframe(future_wages_nn[['year'] + selected_columns])

    return future_wages_nn




