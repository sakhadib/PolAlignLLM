import pandas as pd

# File path
input_file_path = r'D:\Research\BiasClass--Classify-News-Bias-Using-LLM\raw_data\Known_Bias.csv'
output_file_path = r'D:\Research\BiasClass--Classify-News-Bias-Using-LLM\data\labeled_data.csv'

# Load the dataset
df = pd.read_csv(input_file_path)

# Filter: Ensure the 'source_bias' column only contains allowed classes
allowed_biases = ['Far Left', 'Left', 'Lean Left', 'Center', 'Lean Right', 'Right', 'Far Right']
df = df[df['source_bias'].isin(allowed_biases)]

# Save the cleaned data
df.to_csv(output_file_path, index=False)

print("Dataset cleaned and saved as labeled_data.csv.")
