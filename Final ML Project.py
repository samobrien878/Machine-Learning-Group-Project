import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load Data
file_path = 'VA_Traffic_Data_2016_March_2023.csv'
df = pd.read_csv(file_path)

# Function to standardize street names dynamically
def standardize_street_name(street):
    # Remove trailing directions (N, S, E, W)
    street = re.sub(r'\s[NSWE]$', '', street)
    # Remove dashes (e.g., "I-95" -> "I95")
    street = street.replace('-', '')
    # Add additional cleaning rules as necessary (e.g., merging variations)
    return street

# Ensure all values in the Street column are strings and handle missing values
df['Street'] = df['Street'].fillna('Unknown')  # Replace NaN with 'Unknown'
df['Street'] = df['Street'].astype(str)        # Convert all values to strings

# Apply the function to standardize street names
df['Street'] = df['Street'].apply(standardize_street_name)

# Step 1: Preprocessing
street_accident_counts = df.groupby('Street').size().reset_index(name='Accident_Count')
street_accident_counts_sorted = street_accident_counts.sort_values(by='Accident_Count', ascending=False)
top_500_streets = street_accident_counts_sorted.head(150)

filtered_df = df[df['Street'].isin(top_500_streets['Street'])].copy()

filtered_df['Start_Time'] = pd.to_datetime(filtered_df['Start_Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
filtered_df['YearMonth'] = filtered_df['Start_Time'].dt.to_period('M')
filtered_df['Month'] = filtered_df['Start_Time'].dt.month

monthly_accidents = filtered_df.groupby(['Street', 'YearMonth']).size().reset_index(name='Accident_Count')

weather_features = filtered_df.groupby(['Street', 'YearMonth']).agg({
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Pressure(in)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Precipitation(in)': 'sum',
    'Weather_Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Month': 'first'
}).reset_index()

monthly_accidents_with_weather = pd.merge(monthly_accidents, weather_features, on=['Street', 'YearMonth'], how='left')

monthly_accidents_with_weather.dropna(inplace=True)

# Encode categorical variables
monthly_accidents_with_weather['Street_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Street'])
monthly_accidents_with_weather['Weather_Condition_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Weather_Condition'])

# Weather conditions as binary features
weather_types = ['Rain', 'Snow', 'Fog', 'Clear', 'Cloudy']
for weather in weather_types:
    monthly_accidents_with_weather[f'Weather_{weather}'] = monthly_accidents_with_weather['Weather_Condition'].str.contains(weather, case=False, na=False).astype(int)

# Location features
location_features = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                     'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                     'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
                     'Astronomical_Twilight']

# Ensure binary columns are in the DataFrame
for feature in location_features:
    if feature not in monthly_accidents_with_weather.columns:
        monthly_accidents_with_weather[feature] = 0

# Feature selection
features = ['Street_Encoded', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Precipitation(in)', 'Weather_Condition_Encoded', 'Month'] + \
           [f'Weather_{weather}' for weather in weather_types] + location_features

# Step 2: Prepare Data for Regression
X = monthly_accidents_with_weather[features]  # Use same features as before
y = monthly_accidents_with_weather['Accident_Count']  # Use accident count as the target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Regression Model - Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Step 4: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Step 5: Feature Importance Visualization
plt.figure(figsize=(12, 6))
plt.bar(X.columns, regressor.feature_importances_)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest Regressor")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Visualize Predictions for the Top 10 Streets
test_set = monthly_accidents_with_weather.loc[X_test.index].copy()
test_set['Predicted_Accidents'] = y_pred

top_10_streets = test_set.groupby('Street')['Predicted_Accidents'].sum().nlargest(10).index
top_10_data = test_set[test_set['Street'].isin(top_10_streets)]

plt.figure(figsize=(14, 8))
sns.barplot(data=top_10_data, x='Month', y='Predicted_Accidents', hue='Street', ci=None)  # No error bars
plt.title("Top 10 Streets - Predicted Accident Counts by Month")
plt.xlabel("Month")
plt.ylabel("Predicted Accident Count")
plt.xticks(ticks=np.arange(1, 13), labels=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], rotation=45, ha="right")
plt.legend(title="Street", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Evaluate Model Performance on Training Data
y_train_pred = regressor.predict(X_train)

# Calculate Metrics for Training Data
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate Metrics for Test Data (already done earlier)
mae_test = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Print Performance Metrics for Both Training and Test Data
print("Training Data Performance:")
print(f"  Mean Absolute Error (MAE): {mae_train}")
print(f"  Mean Squared Error (MSE): {mse_train}")
print(f"  R² Score: {r2_train}\n")

print("Test Data Performance:")
print(f"  Mean Absolute Error (MAE): {mae_test}")
print(f"  Mean Squared Error (MSE): {mse_test}")
print(f"  R² Score: {r2_test}")

# Visualize Actual vs Predicted for Training Data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, label="Training Data")
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label="Test Data", color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Ideal Fit")
plt.title("Actual vs Predicted Accident Counts (Training vs Test)")
plt.xlabel("Actual Accident Count")
plt.ylabel("Predicted Accident Count")
plt.legend()
plt.tight_layout()
plt.show()



