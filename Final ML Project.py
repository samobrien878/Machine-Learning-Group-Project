import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = 'VA_Traffic_Data_2016_March_2023.csv'
df = pd.read_csv(file_path)

street_accident_counts = df.groupby('Street').size().reset_index(name='Accident_Count')
street_accident_counts_sorted = street_accident_counts.sort_values(by='Accident_Count', ascending=False)
top_1500_streets = street_accident_counts_sorted.head(500)

# Create a copy to avoid SettingWithCopyWarning
filtered_df = df[df['Street'].isin(top_1500_streets['Street'])].copy()

filtered_df['Start_Time'] = pd.to_datetime(filtered_df['Start_Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
filtered_df['YearMonth'] = filtered_df['Start_Time'].dt.to_period('M')
filtered_df['Month'] = filtered_df['Start_Time'].dt.month  # Extract month as a feature

monthly_accidents = filtered_df.groupby(['Street', 'YearMonth']).size().reset_index(name='Accident_Count')

weather_features = filtered_df.groupby(['Street', 'YearMonth']).agg({
    'Temperature(F)': 'mean',
    'Humidity(%)': 'mean',
    'Pressure(in)': 'mean',
    'Visibility(mi)': 'mean',
    'Wind_Speed(mph)': 'mean',
    'Precipitation(in)': 'sum',
    'Weather_Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Month': 'first'  # Add month to weather features
}).reset_index()

monthly_accidents_with_weather = pd.merge(monthly_accidents, weather_features, on=['Street', 'YearMonth'], how='left')

monthly_accidents_with_weather.dropna(inplace=True)

risk_threshold = monthly_accidents_with_weather['Accident_Count'].quantile(0.66)

def classify_risk_binary(count):
    return 1 if count > risk_threshold else 0

monthly_accidents_with_weather['Risk_Class_Binary'] = monthly_accidents_with_weather['Accident_Count'].apply(classify_risk_binary)

monthly_accidents_with_weather['Street_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Street'])
monthly_accidents_with_weather['Weather_Condition_Encoded'] = LabelEncoder().fit_transform(monthly_accidents_with_weather['Weather_Condition'])

weather_types = ['Rain', 'Snow', 'Fog', 'Clear', 'Cloudy']
for weather in weather_types:
    monthly_accidents_with_weather[f'Weather_{weather}'] = monthly_accidents_with_weather['Weather_Condition'].str.contains(weather, case=False, na=False).astype(int)
location_features = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                     'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                     'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
                     'Astronomical_Twilight']

# Ensure binary columns are in the DataFrame
for feature in location_features:
    if feature not in monthly_accidents_with_weather.columns:
        monthly_accidents_with_weather[feature] = 0  # Default to 0 if missing

features = ['Street_Encoded', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Precipitation(in)', 'Weather_Condition_Encoded', 'Month'] + \
           [f'Weather_{weather}' for weather in weather_types] + location_features

X = monthly_accidents_with_weather[features]
y = monthly_accidents_with_weather['Risk_Class_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_rf = clf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Risky"]))

feature_importances = clf.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(X.columns, feature_importances)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest Classifier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test data
y_pred_rf = clf.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Risky"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix for Binary Classification")
plt.show()
