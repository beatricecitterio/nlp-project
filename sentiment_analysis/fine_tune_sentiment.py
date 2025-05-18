import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import datasets
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def main():
    print("Loading training data...")
    train_df = pd.read_csv('labeled_data_cleaned.csv')
    
    print(f"Training data shape before filtering: {train_df.shape}")
    print("Sample of training data:")
    print(train_df.head())
    
    unique_labels = train_df['label_id'].unique()
    print(f"Unique labels in training data: {unique_labels}")
    print(f"Label distribution before filtering: \n{train_df['label_id'].value_counts()}")
    
    if 'unknown' in train_df['label_id'].values:
        train_df = train_df[train_df['label_id'] != 'unknown'].copy()
        print(f"Training data shape after filtering 'unknown' labels: {train_df.shape}")
        print(f"Label distribution after filtering: \n{train_df['label_id'].value_counts()}")
    
    string_to_numeric = {
        '0.0': 0,  # 'Call to Action / Propaganda'
        '1.0': 1,  # 'Critical / Angry'
        '2.0': 2,  # 'Neutral / Informational'
        '3.0': 3   # 'Supportive / Affirmative / Celebratory'
    }
    
    id2label = {
        0: 'Call to Action / Propaganda',
        1: 'Critical / Angry',
        2: 'Neutral / Informational',
        3: 'Supportive / Affirmative / Celebratory'
    }
    
    label2id = {v: k for k, v in id2label.items()}
    
    model_name = "dbmdz/bert-base-italian-uncased"
    
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text_column = 'content' 
    label_column = 'label_id'
    
    train_df[label_column] = train_df[label_column].map(string_to_numeric)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df[text_column], 
        train_df[label_column], 
        test_size=0.1, 
        random_state=seed,
        stratify=train_df[label_column]
    )
    
    train_dataset = Dataset.from_dict({
        'text': train_texts.tolist(),
        'label': train_labels.tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts.tolist(),
        'label': val_labels.tolist()
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    def format_labels(examples):
        examples['labels'] = examples.pop('label')
        return examples
    
    tokenized_train = tokenized_train.map(format_labels)
    tokenized_val = tokenized_val.map(format_labels)
    
    print("Setting up the model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=1,
        overwrite_output_dir=True,
        report_to="none",
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = (predictions == labels).mean()
        
        return {
            'accuracy': accuracy
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Starting fine-tuning on labeled data...")
    train_result = trainer.train()
    print(f"Training completed. Training metrics: {train_result.metrics}")
    
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation metrics: {eval_results}")
    
    print("Saving the model...")
    trainer.save_model("./model_sentiment")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    main()