import pandas as pd
import torch
from transformers import AutoTokenizer

def prepare_phishing_dataset(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create formatted examples based on 'Email Type'
    def format_example(row):
        text = row['Email Text']
        label = row['Email Type']
        
        # Ensure consistent formatting
        return f"### Human: Classify this email as safe or phishing.\n\n{text}\n\n### Assistant: {label}"
    
    # Apply formatting to each row
    df['formatted_text'] = df.apply(format_example, axis=1)
    
    # Print unique email types to verify
    print("Unique Email Types:", df['Email Type'].unique())
    
    return df['formatted_text'].tolist()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b")

# Prepare dataset
dataset = prepare_phishing_dataset('Phishing_Email.csv')

# Optional: Split into train and validation
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset = dataset[int(len(dataset)*0.8):]

print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")