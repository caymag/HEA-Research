import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Select features
numerical_features = ['Cu', 'Al', 'Ag', 'B', 'Be', 'Ca', 'Co', 'Ce', 'Cr', 'Fe', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pr', 'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr', 'Tss (K)', 'tss (h)', 'CR reduction (%)', 'Tag (K)', 'tag (h)']
categorical_features = ['Aging', 'Secondary thermo-mechanical process']
# Load data
df = pd.read_csv('Corrected_Cu_alloys_database.csv')

# Ensure 'Hardness (HV)' column is numeric, and coerce errors to NaN
df['Hardness (HV)'] = pd.to_numeric(df['Hardness (HV)'], errors='coerce')

# Remove rows with missing values for specific columns
df = df.dropna(subset=['Hardness (HV)', 'Yield strength (MPa)', 'Electrical conductivity (%IACS)'], how='all').reset_index(drop=True)

# Calculate Yield Strength if Hardness (HV) is present and update the column
for index, row in df.iterrows():
    if pd.notna(row["Hardness (HV)"]):  # Only calculate if 'Hardness (HV)' is not NaN
        hv_ys = (row["Hardness (HV)"] * 9.807) / 3
        df.at[index, "Yield strength (MPa)"] = hv_ys

# Ensure 'Yield strength (MPa)' is numeric
df['Yield strength (MPa)'] = pd.to_numeric(df['Yield strength (MPa)'], errors='coerce')

# Drop rows that contain NaN in the target or features
df = df.dropna(subset=['Yield strength (MPa)'] + numerical_features + categorical_features)

# Log-transform the target variable to address skewness
y = np.log1p(df['Yield strength (MPa)'].values.reshape(-1, 1))



# Handle categorical data with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df[numerical_features + categorical_features], y, test_size=0.2, random_state=68)

# Define and tune kernel for Gaussian Process
kernel = Matern(nu=2.5, length_scale=2) + RBF(length_scale=1.0) + WhiteKernel(noise_level=1)

# Create a pipeline to scale and transform the features, then apply GPR
gpr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10))
])

# Fit the model
gpr_pipeline.fit(x_train, y_train.ravel())

# Make predictions
y_pred_log = gpr_pipeline.predict(x_test)

# Inverse the log transformation of predictions and test data
y_pred = np.expm1(y_pred_log)  # Transform predictions back to original scale
y_test = np.expm1(y_test)  # Transform test data back to original scale

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Calculate R^2 (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label="Predictions")
plt.plot(y_test, y_test, color='black', label='Parity Line')  # Plot parity line
plt.xlabel("Experimental YS (MPa)")
plt.ylabel("Predicted YS (MPa)")
plt.title("Copper Alloy Yield Strength Prediction (YS)")
plt.legend()
plt.show()
