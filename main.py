import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_imdb_data(data_path):
    """
    Загружает данные из структурированных папок IMDB датасета
    """

    reviews = []
    sentiments = []

    pos_path = os.path.join(data_path, 'pos')
    pos_files = [f for f in os.listdir(pos_path) if f.endswith('.txt')]
    for i, filename in enumerate(pos_files[:2001]):
        if i % 1000 == 0:
            print(f'Обработано: {i}/{len(pos_files)} позитивных отзывов')
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
            sentiments.append(1)

    neg_path = os.path.join(data_path, 'neg')
    neg_files = [f for f in os.listdir(neg_path) if f.endswith('.txt')]
    for i, filename in enumerate(neg_files[:2001]):
        if i % 1000 == 0:
            print(f'Обработано: {i}/{len(neg_files)} негативных отзывов')
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
            sentiments.append(0)

    return pd.DataFrame({'review' : reviews, 'sentiment' : sentiments})


train_path = 'dataset/train'
test_path = 'dataset/test'

print('Загрузка тренировочных данных...')
train_df = load_imdb_data(train_path)
print('Загрузка тестовых данных...')
test_df = load_imdb_data(test_path)

print(f'Тренировочных данных: {len(train_df)}')
print(f'Тестовых данных: {len(test_df)}')

print('\nРаспределение в тренировочных данных:')
print(train_df['sentiment'].value_counts())
print('\nРаспределение в тестовых данных:')
print(test_df['sentiment'].value_counts())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x='sentiment', data=train_df)
plt.title('Train Data Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='sentiment', data=test_df)
plt.title('Test Data Distribution')
plt.tight_layout()
plt.show()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)