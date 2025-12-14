import pandas as pd
import glob
import os

# Path to your CSV files
csv_path = r"C:\Users\SaifEddine\OneDrive\Desktop\Projects\collect my own data\Data\*.csv"

# Read all CSV files
all_files = glob.glob(csv_path)
df_list = []

print(f"Found {len(all_files)} CSV files")

for filename in all_files:
    df = pd.read_csv(filename)
    df_list.append(df)
    print(f"Loaded: {os.path.basename(filename)} - {len(df)} rows")

# Combine all data
combined_df = pd.concat(df_list, ignore_index=True)

# Save combined dataset WITHOUT header 
combined_df.to_csv('my_activity_data.csv', index=False, header=False)

print(f"\n✓ Total samples: {len(combined_df)}")
print(f"✓ Samples per activity:")
print(combined_df.groupby('Label').size())
print(f"\n✓ Saved to: my_activity_data.csv")