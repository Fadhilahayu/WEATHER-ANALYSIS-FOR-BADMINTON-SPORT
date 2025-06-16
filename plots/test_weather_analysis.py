import unittest
import pandas as pd
import numpy as np
import os
from weather_analysis_for_badminton_sport_full import (
    plot_categorical_distribution,
    plot_categorical_relationships,
    train_model,
    plot_confusion_matrix,
    plots_dir
)

class TestWeatherAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use more rows to ensure both class labels are present
        cls.sample_df = pd.DataFrame({
            'Outlook': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Overcast'],
            'Temperature': ['Hot', 'Mild', 'Cool', 'Cool', 'Hot', 'Mild', 'Cool', 'Hot'],
            'Humidity': ['High', 'Normal', 'High', 'High', 'Normal', 'High', 'Normal', 'High'],
            'Wind': ['Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Weak'],
            'Play_Badminton': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
        })
        os.makedirs(plots_dir, exist_ok=True)

    def test_plot_categorical_distribution(self):
        custom_colors = {
            'Outlook': 'lightblue', 'Temperature': 'lightgreen',
            'Humidity': 'lightsalmon', 'Wind': 'lightcoral', 'Play_Badminton': 'lightgray'
        }
        plot_categorical_distribution(self.sample_df, ['Outlook'], colors=custom_colors)
        self.assertTrue(os.path.exists(os.path.join(plots_dir, 'distribution_Outlook.png')))

    def test_plot_categorical_relationships(self):
        plot_categorical_relationships(self.sample_df, ['Outlook', 'Play_Badminton'])
        expected_path = os.path.join(plots_dir, 'relationship_Outlook_vs_Play_Badminton.png')
        self.assertTrue(os.path.exists(expected_path))

    def test_train_model_output(self):
        conf_matrix = train_model(self.sample_df)
        self.assertEqual(conf_matrix.shape, (2, 2))
        self.assertTrue(np.issubdtype(conf_matrix.dtype, np.integer))

    def test_plot_confusion_matrix(self):
        conf_matrix = np.array([[2, 1], [0, 2]])
        plot_confusion_matrix(conf_matrix)
        expected_path = os.path.join(plots_dir, 'confusion_matrix.png')
        self.assertTrue(os.path.exists(expected_path))

if __name__ == '__main__':
    unittest.main()
