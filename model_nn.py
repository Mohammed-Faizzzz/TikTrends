import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv('synthetic_tiktok_data.csv')

# Define the features and target variable
X = df[['Hashtag', 'Caption', 'Location']]
y = df['Likes']

# Preprocessing for categorical data: one-hot encoding
categorical_features = ['Hashtag', 'Caption', 'Location']
one_hot = OneHotEncoder()

# Create a column transformer to transform categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot, categorical_features)],
    remainder='passthrough')

# Apply the preprocessing
X = preprocessor.fit_transform(X)

# Normalize the input features and the target variable
scaler_X = StandardScaler(with_mean=False)
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural network architecture with LeakyReLU activation and regularization
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.01),
    Dense(64, kernel_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.01),
    Dense(1, activation='linear')  # Linear activation for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = np.mean((scaler_y.inverse_transform(y_pred).flatten() - scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()) ** 2)
print(f'Mean Squared Error: {mse}')
baseline_mse = np.mean((scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten() - np.mean(scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()))**2)
print(f'Baseline MSE: {baseline_mse}')
