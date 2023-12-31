import os
import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Step 4: Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples, return_special_tokens_mask=True)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="tinyshakespeare.txt",
    block_size=128
)

# Step 5: Prepare the dataset for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 6: Create a model configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_ctx=128,
    n_embd=768,
    n_layer=12,
    n_head=12,
    num_labels=1,
)

# Step 7: Instantiate the model
model = GPT2LMHeadModel(config)

# Step 8: Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=100,
)

# Step 9: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# Step 10: Save the model and the tokenizer
trainer.save_model("tiny_shakespeare_gpt2")
tokenizer.save_pretrained("tiny_shakespeare_gpt2")

# Step 11: Generate text using the trained model and user input
input_text = input("Please enter a text prompt: ")
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Move the input tensor to the same device as the model
device = model.device
input_ids = input_ids.to(device)

# Use top-k sampling for generating multiple output sequences
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=3,
    do_sample=True,  # Enable sampling
    top_k=50  # Top-k sampling
)

for i, generated_text in enumerate(output):
    print(f"Generated text {i + 1}:")
    print(tokenizer.decode(generated_text))