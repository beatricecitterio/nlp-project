from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Load and tokenize dataset
dataset = load_dataset("json", data_files="tweets_dataset.jsonl", split="train")

model_name = "mistralai/Mistral-7B-v0.1" 
#Alternative models: "cosimoiaia/Loquace-7B-Mistral", "DeepMount00/Mistral-Ita-7b", "mii-community/zefiro-7b-base-ITA", "galatolo/cerbero-7b-openchat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Important for Mistral

# def format_example(example):
#     return f"{example['prompt']} {example['response']}"

def format_example(example):
    return (
        f"Write a tweet.\n"
        f"Topic: {example['topic']}; "
        f"Keywords: {example['keywords']}; "
        f"Party: {example['party']}; "
        f"Sentiment: {example['sentiment']}\n"
        f"Tweet: {example['response']}"
    )

dataset = dataset.map(lambda x: {"text": format_example(x)})

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tweet-model",
    per_device_train_batch_size=2, #for NVIDIA T4 maybe use 1-2
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    learning_rate=2e-4,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
model.save_pretrained("final-tweet-model")
tokenizer.save_pretrained("final-tweet-model")
