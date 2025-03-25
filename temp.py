import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the data

data = pd.read_csv('wages_by_education.csv').convert_dtypes()

for col in data.columns:
    if data[col].dtype == 'int32':
        data[col] = data[col].astype('int64')
    if col != 'year':
        data[col] = data[col].astype(float)
    else:
        data[col] = data[col].astype(int)

if data.isnull().sum().sum() > 0:
    data.dropna(inplace=True)


# Select the columns we're interested in
columns = ['year', 'men_high_school', 'women_high_school', 
           'men_bachelors_degree', 'women_bachelors_degree', 
           'men_advanced_degree', 'women_advanced_degree']

data = data[columns]

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

# Print the best parameters
print("Best parameters for MLPRegressor:", grid_search.best_params_)

# Predict on the testing set using the best MLPRegressor model
y_pred_mlp = grid_search.predict(X_test)

# Calculate and print the evaluation metrics for MLPRegressor
print("MLPRegressor Mean Squared Error:", mean_squared_error(y_test, y_pred_mlp))
print("MLPRegressor R2 Score:", r2_score(y_test, y_pred_mlp))

# Define the Linear Regression model
lr_model = LinearRegression()

# Train the Linear Regression model
lr_model.fit(X_train, y_train)

# Predict on the testing set using the Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Calculate and print the evaluation metrics for Linear Regression
print("Linear Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))

# Create a DataFrame for the next 50 years and normalize the years
future_years = pd.DataFrame({'year': range(2023, 2073)})
scaled_future_years = scaler_x.transform(future_years)

# Predict future wages using the trained models
future_wages_predictions_mlp = grid_search.predict(scaled_future_years)
future_wages_predictions_lr = lr_model.predict(scaled_future_years)

# Inverse transform the predictions to get the actual wage values
future_wages_mlp = pd.DataFrame(scaler_y.inverse_transform(future_wages_predictions_mlp), columns=columns[1:])
future_wages_mlp['year'] = future_years['year']

future_wages_lr = pd.DataFrame(scaler_y.inverse_transform(future_wages_predictions_lr), columns=columns[1:])
future_wages_lr['year'] = future_years['year']

# Display the predicted future wages for MLPRegressor
print("\nPredicted future wages for MLPRegressor:")
print(future_wages_mlp)

# Display the predicted future wages for Linear Regression
print("\nPredicted future wages for Linear Regression:")
print(future_wages_lr)
