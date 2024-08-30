import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

training_data=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_4_DS\archive (4)\twitter_training.csv')
validation=pd.read_csv(r'C:\Users\Student\Desktop\Prodigy_Infotech\Task_4_DS\archive (4)\twitter_validation.csv')

print(training_data.head())
print(training_data.columns)

#Configuring Column Names
training_data.columns=['ID','Brand','Sentiment', 'Tweet']
validation.columns=['ID','Brand','Sentiment', 'Tweet']
print(training_data.columns)

#Null values
print(training_data.isnull().sum())
print(validation.isnull().sum())

#Dropping null values in Tweet Column
training_data=training_data.dropna(subset=['Tweet'])
print(training_data.isnull().sum())
print(training_data.info())


sentiment_counts_training = training_data['Sentiment'].value_counts()
sentiment_counts_validation = validation['Sentiment'].value_counts()

print(sentiment_counts_training)
print(sentiment_counts_validation)

plt.figure(figsize=(10,6))
sns.boxplot(x=sentiment_counts_training.index, y=sentiment_counts_training.values, palette='plasma')
plt.title("Sentiment for Brands")
plt.xlabel("Sentiment")
plt.ylabel("Tweets")
plt.show()

#Countplot for Training Data Set
plt.figure(figsize=(14,8))
sns.countplot(data=training_data, x='Brand', hue='Sentiment', palette='viridis')
plt.title("Sentiment of Brands")
plt.xlabel("Brand")
plt.ylabel("y")
plt.xticks(rotation=45, ha='right')
plt.show()


#Countplot for Validation Data
plt.figure(figsize=(14,8))
sns.countplot(data=validation, x='Brand', hue='Sentiment', palette='plasma')
plt.title("Brands in Validation Set")
plt.xlabel("Brands")
plt.ylabel("Sentiment")
plt.xticks(rotation=45, ha='right')
plt.show()


positive_words=' '.join(training_data[training_data['Sentiment']=='Positive']['Tweet'])
word_count=WordCloud(width=800, height=400, background_color='white').generate(positive_words)

#Mostly used Words in Positive Sentiment
plt.figure(figsize=(14,8))
plt.imshow(word_count, interpolation='bilinear')
plt.title("Mostly used Words in Positive Sentiment")
plt.axis('off')
plt.show()


negative_words=' '.join(training_data[training_data['Sentiment']=='Negative']['Tweet'])
negative_word_count=WordCloud(width=800,height=400,background_color='white').generate(negative_words)

#Mostly used Words in Negative Sentiment
plt.figure(figsize=(14,6))
plt.imshow(negative_word_count, interpolation='bilinear')
plt.title("Mostly used Words in Negative Sentiment")
plt.axis('off')
plt.show()
