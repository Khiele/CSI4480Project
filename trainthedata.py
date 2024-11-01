import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, load_dataset

# Load the pre-trained model
max_seq_length = 512
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Ampere GPUs
load_in_4bit = True  # Quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare the model for training with LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,  # Rank of LoRA adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 8,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

# Function to format the prompt
def format_prompt(text):
    return f"### Human: Classify this email as safe or phishing.\n\n{text}\n\n### Assistant:"

# Load and slice dataset
dataset = load_dataset('csv', data_files='Phishing_Email.csv', split='train')
dataset = dataset.select(range(200))  # Select only the first 200 entries

# Prepare dataset with formatting
def prepare_dataset(examples):
    texts = [
        f"### Human: Classify this email as safe or phishing.\n\n{text}\n\n### Assistant: {label}" 
        for text, label in zip(examples['Email Text'], examples['Email Type'])
    ]
    return {"text": texts}

# Process and split the dataset
processed_dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
processed_dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)

# Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    output_dir = "./phishing_detection_model",
)

# Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = processed_dataset['train'],
    eval_dataset = processed_dataset['test'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = training_args,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./phishing_detection_model")
