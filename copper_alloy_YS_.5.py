import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, kendalltau

# Select features
numerical_features = ['Cu', 'Al', 'Ag', 'B', 'Be', 'Ca', 'Co', 'Ce', 'Cr', 'Fe', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pr', 'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr', 'Tss (K)', 'tss (h)', 'CR reduction (%)', 'Tag (K)', 'tag (h)']
categorical_features = ['Aging', 'Secondary thermo-mechanical process']

# Load data
df = pd.read_csv('Corrected_Cu_alloys_database.csv')

# Ensure 'Hardness (HV)' column is numeric
df['Hardness (HV)'] = pd.to_numeric(df['Hardness (HV)'])

# Remove rows with missing target values
df = df.dropna(subset=['Hardness (HV)', 'Yield strength (MPa)', 'Electrical conductivity (%IACS)'], how='all').reset_index(drop=True)

# Calculate Yield Strength if Hardness (HV) is present and update the column
for index, row in df.iterrows():
    if pd.notna(row["Hardness (HV)"]):  # Only calculate if 'Hardness (HV)' is not NaN
        hv_ys = (row["Hardness (HV)"] * 9.807) / 3
        df.at[index, "Yield strength (MPa)"] = hv_ys

# Ensure 'Yield strength (MPa)' is numeric
df['Yield strength (MPa)'] = pd.to_numeric(df['Yield strength (MPa)'])

# Drop rows that contain NaN in the target or features
df = df.dropna(subset=['Yield strength (MPa)'] + numerical_features + categorical_features)

# Log-transform the target variable to address skewness
y = (df['Yield strength (MPa)'].values.reshape(-1, 1))

# Handle categorical data with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df[numerical_features + categorical_features], y, test_size=0.8, random_state=68)

# Define and tune kernel for Gaussian Process
kernel = RBF(length_scale=0.5) + WhiteKernel()

# Create a pipeline to scale and transform the features, then apply GPR
gpr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True))
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
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, label="Predicted Values", color='blue', edgecolors='k', alpha=0.7)
plt.plot(y_test, y_test, color='black', linestyle='--', label='Parity Line')  # Plot parity line
plt.xlabel("Experimental Yield Strength (YS) [MPa]", fontsize=14)
plt.ylabel("Predicted Yield Strength (YS) [MPa]", fontsize=14)
plt.title("Yield Strength Prediction for Copper Alloys (0.50 LS)", fontsize=16)
plt.text(0.05, 0.85, f'MAE: {mae:.2f} MPa', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.80, f'τ: {tau:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.75, f'ρ: {spearman:.2f}', transform=plt.gca().transAxes, fontsize=12) 
plt.xlim(np.min(y_test), np.max(y_test))
plt.ylim(np.min(y_pred),np.max(y_pred))   
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()