import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("eliasmarcon/luna-16/data/LUNA_16/annotations.csv")

annotations_df = pd.read_csv(path)

print("Path to dataset files:", path)

print(annotations_df.head())