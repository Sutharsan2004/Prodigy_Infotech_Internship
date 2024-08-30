import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_3_DS\bank+marketing\bank\bank.csv', sep=';')
print("Using bank.csv Dataset\n")
print("Data Samples :\n",df.head())
print("\n")
print("Columns Are...\n",df.columns)
print("\n")


#Exploratory Data Analysis

print("Info :\n",df.info())
print("Checking Null Count\n",df.isnull().sum())

#Visualization of Age

plt.hist(df['age'], bins=20, edgecolor='black')
plt.title("age")
plt.xlabel("age")
plt.ylabel("freq")
plt.show()

#Categorizing Age into 4 groups
"""
bins=[18, 25, 35, 55, 80]
labels=['Adults', 'Aged Adult', 'Middle Aged', 'Senior Citizens']
df['agegroup']=pd.cut(df['age'], bins=bins, labels=labels)

age_bal=df.groupby('agegroup',observed=True)['balance'].mean()
print(age_bal)

sns.histplot(df['agegroup'], bins=10, edgecolor='black')
plt.title("Age Groups")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
"""

#Converting Categorical Variables into Numerical
category_columns=df.select_dtypes(include='object').columns
print("\nCategory variable\n", category_columns)
print("\n")

df_encode=pd.get_dummies(df, columns=category_columns, drop_first=True)


#X - Independant Variable Which Contains the Features.. So we removed the target column 'y_yes' from X.
#y - Dependant Variable Which contains the target.. So we added 'y_yes'.

X=df_encode.drop('y_yes', axis=1)#Axis = 1 represnt the column
y=df_encode['y_yes']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model=DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred)*100)
print("\n")
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
print("\n")
print("Classification :\n ", classification_report(y_test, y_pred))




