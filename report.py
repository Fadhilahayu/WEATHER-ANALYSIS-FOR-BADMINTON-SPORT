# -*- coding: utf-8 -*-
"""
Generate MNE HTML report for Weather Analysis for Badminton Sport
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mne  # pip install mne

# Set your project base path
base_path = r'C:\Users\User\IdeaProjects\WEATHER-ANALYSIS-FOR-BADMINTON-SPORT\plots'

# List of plot image filenames (adjust as needed)
image_files = [
    ('distribution_Outlook.png', 'Outlook Distribution'),
    ('distribution_Temperature.png', 'Temperature Distribution'),
    ('distribution_Humidity.png', 'Humidity Distribution'),
    ('distribution_Wind.png', 'Wind Distribution'),
    ('distribution_Play_Badminton.png', 'Play Badminton Distribution'),
    ('relationship_Outlook_vs_Play_Badminton.png', 'Outlook vs Play Badminton'),
    ('relationship_Temperature_vs_Play_Badminton.png', 'Temperature vs Play Badminton'),
    ('relationship_Humidity_vs_Play_Badminton.png', 'Humidity vs Play Badminton'),
    ('relationship_Wind_vs_Play_Badminton.png', 'Wind vs Play Badminton'),
    ('confusion_matrix.png', 'Confusion Matrix')
]

# Create the MNE report
report = mne.Report(title='Weather Analysis for Badminton Sport')

# Function to add image to the report
def add_image_to_report(img_file, title, section='Data Visualization'):
    img_path = os.path.join(base_path, img_file)
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)
        report.add_figure(fig=fig, title=title, section=section)
        plt.close(fig)
    else:
        print(f"[❌] File not found: {img_path}")

# Add all images
for img_file, title in image_files:
    add_image_to_report(img_file, title)

# Save the report
report_file = os.path.join(base_path, 'weather_analysis_report.html')
report.save(fname=report_file, overwrite=True)

print(f"✅ Report successfully created and saved at:\n{report_file}")
