import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv('synthetic_tiktok_data.csv')

# Define the features and target variable
X = df[['Hashtag', 'Caption', 'Location']]
y = df['Likes']  # Example: Predicting likes

# Preprocessing for categorical data: one-hot encoding
categorical_features = ['Hashtag', 'Caption', 'Location']
one_hot = OneHotEncoder()

# Create a column transformer to transform categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot, categorical_features)],
    remainder='passthrough')

# # Create a pipeline that encapsulates preprocessing and modeling
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('regressor', LinearRegression())])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# Adjust the pipeline to use a Random Forest regressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
