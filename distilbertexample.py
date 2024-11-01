# Install required libraries
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Set the seed for reproducibility
torch.manual_seed(42)

# Load your dataset using Pandas
df = pd.read_csv('Phishing_Email.csv')

# Take the first 50 samples
df = df.head(30000)

# Check for imbalance in the dataset
print("Label distribution:")
print(df['Email Type'].value_counts())

# Check for NaN values in the relevant columns
print("NaN values before filling:")
print(df[['Email Text', 'Email Type']].isnull().sum())

# Fill NaN values
df['Email Text'].fillna('', inplace=True)  # Replace with empty string
df['Email Type'].fillna('Unknown', inplace=True)  # Replace with an appropriate label

# Check again for NaN values after filling
print("\nNaN values after filling:")
print(df[['Email Text', 'Email Type']].isnull().sum())

# Create a custom dataset class to handle text data
class PhishingEmailDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        # Extract the email text and type from the DataFrame
        email_text = self.df.iloc[idx]['Email Text']
        label = 1 if self.df.iloc[idx]['Email Type'] == 'Phishing Email' else 0  # Convert to integer for classification

        # Check for NaN in email_text
        if pd.isna(email_text):
            raise ValueError(f"NaN detected in email_text at index {idx}.")

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            email_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Preprocess the input IDs and attention mask
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Convert labels to tensor
        labels = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }

    def __len__(self):
        return len(self.df)

# Load the tokenizer and set up hyperparameters
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

max_len = 512
batch_size = 16
num_epochs = 5  # Adjust this value as needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset and data loader
dataset = PhishingEmailDataset(df, tokenizer, max_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the pre-trained model and move it to the device
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
