# -*- coding: utf-8 -*-
"""
Career Predictor using Random Forest Classifier
Author: Suhani Dwivedi
Dataset: cs_students.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cs_students.csv')
df = df.drop(['Student ID', 'Major'], axis=1)

career_counts = df['Future Career'].value_counts()
rare_careers = career_counts[career_counts <= 5].index
df['Future Career'] = df['Future Career'].apply(lambda x: 'Other' if x in rare_careers else x)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'GPA']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Interested Domain'])
    ])

X = df.drop('Future Career', axis=1)
y = df['Future Career'].astype(str)
X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(pd.DataFrame(cm, index=model.classes_, columns=model.classes_),
            annot=True, fmt='d', cmap='Blues')
plt.show()
