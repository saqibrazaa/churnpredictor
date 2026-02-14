import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('churn.csv')

# Display the first few rows of the dataset
print(df.head())