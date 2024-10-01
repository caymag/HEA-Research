import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Select features
numerical_features = ['Cu', 'Al', 'Ag', 'B', 'Be', 'Ca', 'Co', 'Ce', 'Cr', 'Fe', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pr', 'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr', 'Tss (K)', 'tss (h)', 'CR reduction (%)', 'Tag (K)', 'tag (h)']
categorical_features = ['Aging', 'Secondary thermo-mechanical process']

# Load data
df = pd.read_csv('Corrected_Cu_alloys_database.csv')

# Remove rows with missing values for specific columns
df = df.dropna(subset=['Hardness (HV)', 'Yield strength (MPa)', 'Electrical conductivity (%IACS)'], how='all').reset_index(drop=True)

# Calculate Yield Strength if Hardness (HV) is present and update the column
for index, row in df.iterrows():
    if pd.notna(row["Hardness (HV)"]):  
        hv_ys = (row["Hardness (HV)"] * 9.807) / 3
        df.at[index, "Yield strength (MPa)"] = hv_ys

# Drop rows that contain NaN in the target or features
df = df.dropna(subset=['Yield strength (MPa)'] + numerical_features + categorical_features)

# Obtain target data
y = (df['Yield strength (MPa)'].values.reshape(-1, 1))


# Handle categorical data with OneHotEncoder and continuous data with StandardScalar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df[numerical_features + categorical_features], y, test_size=0.8, random_state=68)

# Define and tune kernel for Gaussian Process
kernel = Matern(nu=2.5, length_scale=2) + WhiteKernel(noise_level=1)

# Create a pipeline to scale and transform the features, then apply GPR
gpr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True))
])

# Fit the model
gpr_pipeline.fit(x_train, y_train.ravel())

# Make predictions
y_pred = gpr_pipeline.predict(x_test)

# Calculate and display errors and metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
spearman, _ = spearmanr(y_test, y_pred)
tau, _ = kendalltau(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')
print(f'Spearman: {spearman:.2f}')  
print(f'Tau: {tau:.2f}') 

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label="Predicted Values", color='blue', edgecolors='k', alpha=0.7)
plt.plot(y_test, y_test, color='black', linestyle='--', label='Parity Line')  # Plot parity line
plt.xlabel("Experimental Yield Strength (YS) [MPa]", fontsize=14)
plt.ylabel("Predicted Yield Strength (YS) [MPa]", fontsize=14)
plt.title("Yield Strength Prediction for Copper Alloys", fontsize=16)
plt.text(0.05, 0.80, f'MAE: {mae:.2f} MPa', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.75, f'τ: {tau:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.70, f'r2: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12)  
plt.text(0.05, 0.65, f'rmse: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.60, f'ρ: {spearman:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
