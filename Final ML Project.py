import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re

file_path = 'VA_Traffic_Data_2016_March_2023.csv'
df = pd.read_csv(file_path)

#standardizing street names dynamically
def standardize_street_name(street):
    street = re.sub(r'\s[NSWE]$', '', street) 
    street = street.replace('-', '')
    return street

df['Street'] = df['Street'].fillna('Unknown').astype(str).apply(standardize_street_name)

#preprocessing
street_accident_counts = df.groupby('Street').size().reset_index(name = 'Accident_Count')
street_accident_counts_sorted = street_accident_counts.sort_values(by = 'Accident_Count', ascending = False)
top_150_streets = street_accident_counts_sorted.head(150)

filtered_df = df[df['Street'].isin(top_150_streets['Street'])].copy()

filtered_df['Start_Time'] = pd.to_datetime(filtered_df['Start_Time'], format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
filtered_df['Month'] = filtered_df['Start_Time'].dt.month

#data aggregation
monthly_accidents = filtered_df.groupby(['Street', 'Month']).size().reset_index(name = 'Accident_Count')

weather_features = filtered_df.groupby(['Street', 'Month']).agg({
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Pressure(in)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Precipitation(in)': 'sum',
    'Weather_Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}).reset_index()

#merge accident data and weather features
monthly_accidents_with_weather = pd.merge(monthly_accidents, weather_features, on = ['Street', 'Month'], how = 'left')
monthly_accidents_with_weather.dropna(inplace = True)

monthly_accidents_with_weather['Street_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Street'])
monthly_accidents_with_weather['Weather_Condition_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Weather_Condition'])

#binary features for weather conditions
weather_types = ['Rain', 'Snow', 'Fog', 'Clear', 'Cloudy']
for weather in weather_types:
    monthly_accidents_with_weather[f'Weather_{weather}'] = monthly_accidents_with_weather['Weather_Condition'].str.contains(weather, case = False, na = False).astype(int)

#location features
location_features = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                     'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                     'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
                     'Astronomical_Twilight']

for feature in location_features:
    if feature not in monthly_accidents_with_weather.columns:
        monthly_accidents_with_weather[feature] = 0

#feature selection
features = ['Street_Encoded', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Precipitation(in)', 'Weather_Condition_Encoded', 'Month'] + \
           [f'Weather_{weather}' for weather in weather_types] + location_features

X = monthly_accidents_with_weather[features]
y = monthly_accidents_with_weather['Accident_Count']

#triain-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#simplified random forest regressor
regressor = RandomForestRegressor(n_estimators = 100, max_depth = 20, random_state = 42)
regressor.fit(X_train, y_train)
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

#put predictions in df
test_set = monthly_accidents_with_weather.loc[X_test.index].copy()
test_set['Predicted_Accidents'] = y_pred_test

#get predictions by month
monthly_predictions = test_set.groupby('Month')['Predicted_Accidents'].sum().reset_index()

#model prediction
plt.figure(figsize = (10, 6))
sns.barplot(data = monthly_predictions, x = 'Month', y = 'Predicted_Accidents', palette = 'viridis')
plt.title("What the Model Predicts: Predicted Accident Counts by Month", fontsize = 16)
plt.xlabel("Month", fontsize = 14)
plt.ylabel("Predicted Accident Counts", fontsize = 14)
plt.xticks(ticks = np.arange(12), labels = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], rotation = 45, ha = "right")
plt.tight_layout()
plt.show()

#predicted vs actual
plt.figure(figsize = (10, 6))
sns.scatterplot(x = y_test, y = y_pred_test, alpha = 0.7, color = "blue", label = "Data Points")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = 'red', linestyle = '--', label = "Perfect Prediction")
plt.title("Predicted vs Actual Accident Counts", fontsize = 16)
plt.xlabel("Actual Accident Counts", fontsize = 14)
plt.ylabel("Predicted Accident Counts", fontsize = 14)
plt.legend()
plt.tight_layout()
plt.show()

#feature importance visualization
plt.figure(figsize = (10, 6))
sorted_idx = np.argsort(regressor.feature_importances_)
plt.barh(X.columns[sorted_idx], regressor.feature_importances_[sorted_idx], color = 'skyblue')
plt.xlabel("Feature Importance", fontsize = 14)
plt.ylabel("Features", fontsize = 14)
plt.title("Feature Importance in Random Forest Regressor", fontsize = 16)
plt.tight_layout()
plt.show()

#metrics, insights
print(f"Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
print(f"MAE = {mae:.4f}, MSE = {mse:.4f}")
