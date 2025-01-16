# Country-Economy Analysis
Country Economy 2000-2023:  USA, Canada, Australia, Russia, India, China



Machine Learning outcomes: Model: GradientBoostingRegressor has highest accuracy R-squared: 0.977106391701919 & Mean Absolute Error (MAE): 0.1449901012841944
Recommend adoption to predicting Unemployment rate versus variables as follows:

# Unemployment rate versus Relevant features
X = df[['Poverty Rate (%)', 'Inflation Rate (%)', 'Literacy Rate (%)']]
y = df['Unemployment Rate (%)']


Model: LinearRegression
R-squared: 0.7836333933350148
Mean Squared Error (MSE): 0.333076143574366
Root Mean Squared Error (RMSE): 0.5771274933447254
Mean Absolute Error (MAE): 0.4457972722064925
--------------------


![image](https://github.com/user-attachments/assets/bf7b627b-4031-44c3-afc4-0b034c11e589)



Model: RandomForestRegressor
R-squared: 0.9861173626769679
Mean Squared Error (MSE): 0.021371021034482802
Root Mean Squared Error (RMSE): 0.14618830676385441
Mean Absolute Error (MAE): 0.12467931034482742
--------------------


![image](https://github.com/user-attachments/assets/9ea6c37d-d91e-419d-b52a-faef86476ae0)



Model: GradientBoostingRegressor
R-squared: 0.977106391701919
Mean Squared Error (MSE): 0.03524256761226419
Root Mean Squared Error (RMSE): 0.18773003918463393
Mean Absolute Error (MAE): 0.1449901012841944
--------------------



![image](https://github.com/user-attachments/assets/b68154cb-838b-489c-b7d5-15a208899a1f)




Model: SVR
R-squared: 0.730978320390505
Mean Squared Error (MSE): 0.41413370095954843
Root Mean Squared Error (RMSE): 0.6435322066218196
Mean Absolute Error (MAE): 0.5539305556878131
--------------------


![image](https://github.com/user-attachments/assets/47293346-eb56-43ac-a4b8-bb15f92e63f0)





Machine learning Python Codes:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data (assuming 'country.csv' is in the same directory)
df = pd.read_csv("country.csv") 

# Select relevant features
X = df[['Poverty Rate (%)', 'Inflation Rate (%)', 'Literacy Rate (%)']]
y = df['Unemployment Rate (%)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train, evaluate, and plot 
def train_and_evaluate(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model: {type(model).__name__}")
    print(f"R-squared: {r2}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print("-"*20)

    # Create a suitable plot for each model
    plt.figure(figsize=(8, 6))
    if isinstance(model, LinearRegression): 
        # For Linear Regression, plot predicted vs. actual with regression line
        plt.scatter(y_test, y_pred, alpha=0.5, label='Actual vs. Predicted')
        plt.plot(y_test, y_test, color='red', label='Regression Line') 
        plt.xlabel("Actual Unemployment Rate")
        plt.ylabel("Predicted Unemployment Rate")
        plt.title(f"{type(model).__name__} Predictions")
        plt.legend()
    elif isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        # For tree-based models, use residual plot
        plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
        plt.xlabel("Predicted Unemployment Rate")
        plt.ylabel("Residuals")
        plt.title(f"{type(model).__name__} Residual Plot")
    elif isinstance(model, SVR):
        # For SVR, plot predicted vs. actual
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Unemployment Rate")
        plt.ylabel("Predicted Unemployment Rate")
        plt.title(f"{type(model).__name__} Predictions") 
    plt.show()

# Train and evaluate different models
models = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(random_state=42),
    SVR()
]

for model in models:
    train_and_evaluate(model)

# Create a feature importance plot (for Random Forest)
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance in Predicting Unemployment Rate')
    plt.show()
