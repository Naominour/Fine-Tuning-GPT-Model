import os
import pandas as pd

# Set the directory where the CSV files are stored
files_directory = r'G:\My Drive\Medical_LLM\input_data'

# Function to convert CSV data into a structured text format
def csv_to_txt(df):
    text_data = ""
    for _, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        text_data += f"Question: {question}\nAnswer: {answer}\n\n"
    return text_data

# Initialize an empty string to hold all text data
all_data_txt = ""

# Loop through all files in the directory
for filename in os.listdir(files_directory):
    if filename.endswith(".csv"):  # Process only CSV files
        filepath = os.path.join(files_directory, filename)
        df = pd.read_csv(filepath)
        df = df[['Question', 'Answer']]  # Corrected column name
        all_data_txt += csv_to_txt(df)

# Specify the output directory and file path
output_dir = r'G:\My Drive\Medical_LLM\output_data'
output_txt_path = os.path.join(output_dir, 'Medical_QA_Dataset.txt')

# Ensure the output directory exists, create it if it doesn't
os.makedirs(output_dir, exist_ok=True)  

# Write the concatenated text data to a .txt file using utf-8 encoding
with open(output_txt_path, "w", encoding="utf-8") as text_file:
    text_file.write(all_data_txt)

print("Text file has been successfully generated!")
