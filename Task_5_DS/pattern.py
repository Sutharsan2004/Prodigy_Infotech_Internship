import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_5_DS\archive (4)\US_Accidents_March23.csv')

#print(df.head())
print(df.columns)
df.dropna(inplace=True)
#print(df.isnull().sum())

# Handle missing values
df.dropna(inplace=True)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')


# Extract hour, day of the week, and month
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month

hourly_accident=df.groupby(df['Hour']).size()
"""
hourly_accident.plot(kind='bar')
plt.title("Hourly accidents")
plt.xlabel("Hours")
plt.ylabel("Number of Accidents")
plt.show()

weather_accidents=df.groupby(df['Weather_Condition']).size().sort_values(ascending=False)

weather_accidents.plot(kind='bar')
plt.title("Weather Related accidents")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")
plt.show()

# Assuming that wet, icy, and snowy conditions are bad road conditions
road_condition_mapping = {
    'Clear': 'Good',
    'Cloudy': 'Good',
    'Fog': 'Bad',
    'Rain': 'Bad',
    'Snow': 'Bad',
    'Ice': 'Bad',
}

df['Road_Condition'] = df['Weather_Condition'].map(road_condition_mapping)

# Group by road condition and count accidents
road_accidents = df.groupby('Road_Condition').size().sort_values(ascending=False)

# Plot accidents by road conditions
plt.figure(figsize=(12, 8))
road_accidents.plot(kind='bar', color='orange')
plt.title('Accidents by Road Condition')
plt.xlabel('Road Condition')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.show()



# Numerical features correlation matrix
correlation_matrix = df[['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Severity']].corr()

import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Group by weather condition and severity
severity_weather = df.groupby(['Weather_Condition', 'Severity']).size().unstack()

# Plot accident severity by weather condition
severity_weather.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
plt.title('Accident Severity by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.show()
"""



X = df[['Hour', 'DayOfWeek', 'Month', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']]
y = df['Severity']

# Handle any missing values in X

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))