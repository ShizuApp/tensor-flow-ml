import pandas as pd
df = pd.read_csv('data.csv')

# Standardizing 
df["height_standard"] = (df["height"] - df["height"].mean()) / df["height"].std()

# Normalizing
df["height_normal"] =   \
    (df["height"] - df["height"].min()) / (df["height"].max() - df['height'].min())