# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
print("Current Working Directory:", os.getcwd())

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Step 1: Load the dataset
csv_file = 'Data/medical_examination.csv'

# Ensure the file exists before trying to read it
if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"The file '{csv_file}' was not found. Please ensure it is in the same directory as this script.")

df = pd.read_csv(csv_file)

# Step 2: Add 'overweight' column (BMI > 25 is overweight)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# Step 3: Normalize 'cholesterol' and 'gluc' values
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)  # 0 = normal, 1 = abnormal
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)  # 0 = normal, 1 = abnormal

# Step 4: Categorical plot
df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
fig = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count")
fig.set_axis_labels("Feature", "Total Count")
plt.savefig('catplot.png')
plt.show()

# Step 5: Clean the data
df_cleaned = df[
    (df['ap_lo'] <= df['ap_hi']) &  # diastolic <= systolic
    (df['height'] >= df['height'].quantile(0.025)) &  # height >= 2.5th percentile
    (df['height'] <= df['height'].quantile(0.975)) &  # height <= 97.5th percentile
    (df['weight'] >= df['weight'].quantile(0.025)) &  # weight >= 2.5th percentile
    (df['weight'] <= df['weight'].quantile(0.975))    # weight <= 97.5th percentile
]

# Step 6: Heatmap of the correlation matrix
corr = df_cleaned.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, cbar_kws={'shrink': .5}, ax=ax)
plt.savefig('heatmap.png')
plt.show()
