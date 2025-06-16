# -*- coding: utf-8 -*-
"""WEATHER ANALYSIS FOR BADMINTON SPORT"""

import kagglehub
import os
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

# Global plot directory
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

def download_data():
    print("--- Downloading Dataset ---")
    dataset_path = kagglehub.dataset_download('aditya0kumar0tiwari/play-badminton')
    full_file_path = os.path.join(dataset_path, 'badminton_dataset.csv')
    df = pd.read_csv(full_file_path)
    print("✅ Dataset loaded successfully.")
    return df

def plot_categorical_distribution(df, columns, colors=None):
    for col in columns:
        if col in df.columns:
            plt.figure()
            df[col].value_counts().plot(kind='bar', color=colors.get(col, 'skyblue'))
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'distribution_{col}.png'))
            plt.show()

def plot_categorical_relationships(df, columns):
    for col1, col2 in combinations(columns, 2):
        if col1 in df.columns and col2 in df.columns:
            plt.figure(figsize=(6, 4))
            crosstab = pd.crosstab(df[col1], df[col2], normalize='index') * 100
            sns.heatmap(crosstab, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Heatmap of {col1} vs {col2}')
            plt.xlabel(col2)
            plt.ylabel(col1)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'relationship_{col1}_vs_{col2}.png'))
            plt.show()

def train_model(df):
    df_encoded = pd.get_dummies(df)
    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = BernoulliNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.2f}")

    return conf_matrix

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.show()

def main():
    df = download_data()

    # Categorical distribution plots
    print("\n--- Generating Distribution Plots ---")
    custom_colors = {
        'Outlook': 'lightblue', 'Temperature': 'lightgreen', 'Humidity': 'lightsalmon',
        'Wind': 'lightcoral', 'Play_Badminton': 'lightgray'
    }
    plot_categorical_distribution(df, ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play_Badminton'], custom_colors)

    # Categorical relationships
    print("\n--- Generating Relationship Heatmaps ---")
    for feature in ['Outlook', 'Temperature', 'Humidity', 'Wind']:
        plot_categorical_relationships(df, [feature, 'Play_Badminton'])

    # Train model and plot confusion matrix
    conf_matrix = train_model(df)
    print("\n--- Plotting Confusion Matrix ---")
    plot_confusion_matrix(conf_matrix)

    print(f"\n✅ All plots saved successfully in the '{plots_dir}' folder.")

# Run everything
if __name__ == '__main__':
    main()
