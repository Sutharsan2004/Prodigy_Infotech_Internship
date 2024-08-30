import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_1_DS\Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3152601.csv')

# Plot histogram for 'region'
sns.histplot(df['Region'], discrete=True)

# Add title and labels
plt.title('Histogram of Regions')
plt.xlabel('Region')
plt.ylabel('Frequency')

# Show plot
plt.show()


sns.histplot(df['IncomeGroup'], discrete=True)

plt.title("Histogram for Income")
plt.xlabel("Income")
plt.ylabel("Freq")

plt.show()


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot for 'region'
sns.countplot(x='Region', data=df, ax=axes[0])
axes[0].set_title('Count of Regions')
axes[0].set_xlabel('Region')
axes[0].set_ylabel('Frequency')

# Plot for 'incomegroup'
sns.countplot(x='IncomeGroup', data=df, ax=axes[1])
axes[1].set_title('Count of Income Groups')
axes[1].set_xlabel('Income Group')
axes[1].set_ylabel('Frequency')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
