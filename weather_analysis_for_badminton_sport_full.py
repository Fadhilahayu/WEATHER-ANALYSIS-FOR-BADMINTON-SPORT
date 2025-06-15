# -*- coding: utf-8 -*-
"""WEATHER ANALYSIS FOR BADMINTON SPORT"""

import kagglehub
import os
# **MODIFICATION 1**: Create a directory to save the plots. This runs only once.
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Data downloading remains the same...
aditya0kumar0tiwari_play_badminton_path = kagglehub.dataset_download('aditya0kumar0tiwari/play-badminton')
print('Data source import complete.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from warnings import filterwarnings
filterwarnings('ignore')

# Load the dataset
full_file_path = os.path.join(aditya0kumar0tiwari_play_badminton_path, 'badminton_dataset.csv')
df = pd.read_csv(full_file_path)

# ... (Your EDA and data cleaning comments are here) ...

### `Making a function for Using Bar plots to visualize the distribution of categorical variables`
def plot_categorical_distribution(df, columns, colors=None, background_color=None, foreground_color=None):
    # ... (function setup code is the same) ...
    for col in columns:
        if col in df.columns:
            plt.figure() # Create a new figure
            # ... (your plotting code is the same) ...
            df[col].value_counts().plot(kind='bar', color=colors.get(col))
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

            # **MODIFICATION 2**: Save the current plot to a file in the 'plots' directory
            plt.savefig(os.path.join(plots_dir, f'distribution_{col}.png'))

            plt.show() # Display the plot after saving

custom_colors = {
    'Outlook': 'lightblue', 'Temperature': 'lightgreen', 'Humidity': 'lightsalmon',
    'Wind': 'lightcoral', 'Play_Badminton': 'lightgray'
}

# Now, when you call this function, it will automatically save the plots
print("--- Generating and Saving Distribution Plots ---")
plot_categorical_distribution(df, ['Outlook'], colors=custom_colors, background_color='#FFEFD5', foreground_color='black')
plot_categorical_distribution(df, ['Temperature'], colors=custom_colors, background_color='#FFEFD5', foreground_color='black')
plot_categorical_distribution(df, ['Humidity'], colors=custom_colors, background_color='#FFEFD5', foreground_color='black')
plot_categorical_distribution(df, ['Wind'], colors=custom_colors, background_color='#FFEFD5', foreground_color='black')
plot_categorical_distribution(df, ['Play_Badminton'], colors=custom_colors, background_color='#FFEFD5', foreground_color='black')


# ... (Your chi-square test code is here) ...

### `making Function for finding relationships between columns`
def plot_categorical_relationships(df, columns):
    for col1, col2 in combinations(columns, 2):
        if col1 in df.columns and col2 in df.columns:
            plt.figure(figsize=(6, 4))
            crosstab = pd.crosstab(df[col1], df[col2], normalize='index') * 100
            sns.heatmap(crosstab, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Heatmap of {col1} vs {col2}')
            plt.xlabel(col2)
            plt.ylabel(col1)

            # **MODIFICATION 3**: Save the heatmap to a file
            plt.savefig(os.path.join(plots_dir, f'relationship_{col1}_vs_{col2}.png'))

            plt.show()

# This function will now also save its plots
print("\n--- Generating and Saving Relationship Plots ---")
columns_to_plot = ['Outlook', 'Play_Badminton']
plot_categorical_relationships(df, columns_to_plot)
columns_to_plot = ['Temperature', 'Play_Badminton']
plot_categorical_relationships(df, columns_to_plot)
columns_to_plot = ['Humidity', 'Play_Badminton']
plot_categorical_relationships(df, columns_to_plot)
columns_to_plot = ['Wind', 'Play_Badminton']
plot_categorical_relationships(df, columns_to_plot)


# ... (Your model training code is here) ...
df_encoded = pd.get_dummies(df)
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
bnb_classifier = BernoulliNB()
bnb_classifier.fit(X_train, y_train)
y_pred = bnb_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
# ... (Your evaluation metrics printouts are here) ...


# --- SAVING THE FINAL CONFUSION MATRIX PLOT ---
print("\n--- Generating and Saving Confusion Matrix Plot ---")
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# **MODIFICATION 4**: Save the confusion matrix plot to a file
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))

plt.show()

print(f"\n✅ All plots have been saved successfully in the '{plots_dir}' folder.")