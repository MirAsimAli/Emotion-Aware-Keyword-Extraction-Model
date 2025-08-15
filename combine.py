import pandas as pd

# Replace these with the paths to your CSV files
file1 = "D:/reddit/reddit_anxiety_10k.csv"
file2 = "D:/reddit/reddit_anxiety_2.csv"
file3 = "D:/reddit/reddit_anxiety_3.csv"

# Load the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Combine the DataFrames, aligning columns and filling missing values with NaN
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_file.csv', index=False)

print(f"CSV files '{file1}', '{file2}', and '{file3}' have been combined and saved as 'combined_file.csv'")
