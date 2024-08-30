import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_2_DS\titanic\train.csv')
df.isnull().sum()
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns='Cabin', inplace=True)

df1=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_2_DS\titanic\test.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)

#df.describe()
df.info()
df1.info()

x=df['Age']
y=df['Fare']
plt.scatter(x,y,color='red', marker='o')
plt.title("Scatter plot for Age and Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

#For Age Group

plt.hist(df['Age'], bins=10, edgecolor='black')
plt.title("Age Group of Titanic")
plt.xlabel("Age")
plt.ylabel("freq")
plt.show()

# Categorize ages into groups

print("Categorize ages into groups")
bins=[0, 13, 19, 35, 50, 80]
labels = ['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior']
df['AgeGroup']=pd.cut(df['Age'], bins=bins, labels=labels)

sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins =20)
plt.title("Age Distribution of Survivors vs Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


print("Total number of survived passengers " , df['Survived'].sum())
print("\n")

#Calculating Survivors based on AgeGroup and Pclass

age_class=df.groupby(['AgeGroup','Pclass'],observed=False)['Survived'].mean().unstack()
print("Calculating Survivors based on AgeGroup and Pclass")
print(age_class)
print("\n")


#Calculating Survivors based on AgeGroup and Gender

age_gender_class=df.groupby(['AgeGroup','Sex'], observed=False)['Survived'].mean()
print("Calculating Survivors based on AgeGroup and Gender")
print(age_gender_class)
print("\n")


# Calculate survival rate for each age group
age_group_survival = df.groupby('AgeGroup', observed=False)['Survived'].mean()
print(age_group_survival)
print("\n")


#Fare Calculation by using Gender
fare_rate=df.groupby('Sex')['Fare'].mean()
print("Fare Calculation by using Gender")
print(fare_rate)
print("\n")

#Survival Rate based on Passenger Class
survival_rate=df.groupby('Pclass')['Survived'].mean()
print("Survival Rate based on Passenger Class")
print(survival_rate)
print("\n")

#Survival Based On Genders
gender_survival=df.groupby('Sex')['Survived'].mean()
print("Survival Based On Genders")
print(gender_survival)
print("\n")

#Fare Classification based on PClass
fare_calc=df.groupby('Pclass')['Fare'].mean()
print("Fare Classification based on PClass")
print(fare_calc)
print("\n")

#Survival Rate Based on Age
survival_age=df.groupby('Survived')['Age'].mean()
print("Survival Rate Based on Age")
print(survival_age)
print("\n")

#Pclass Age Group
p_calc=df.groupby('Pclass')['Age'].mean()
print("Pclass Age Group")
print(p_calc)
