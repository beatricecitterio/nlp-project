import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    num_labels = 4  # 0, 1, 2, 3
    
    model_path = "./model_sentiment"
    
    print("Loading pretrained model and tokenizer...") 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    trainer = Trainer(model=model)
    
    print("Loading test data...")
    test_df = pd.read_csv('350m_fine_new_cleaned.csv') # INPUT FILE WE WANT TO TEST
    
    print(f"Test data shape: {test_df.shape}")
    print("Sample of test data:")
    print(test_df.head())
    
    print("Converting labels to integers...")
    
    def convert_to_int(label):
        """Convert various label formats to integers 0-3"""
        try:
            if isinstance(label, str):
                if '.' in label:
                    return int(float(label))
                return int(label)
            elif isinstance(label, (int, float)):
                return int(label)
        except:
            pass
        return None  
    
    test_df['label_numeric'] = test_df['label_id'].apply(convert_to_int)
    
    print("Label distribution after conversion:")
    print(test_df['label_numeric'].value_counts(dropna=False))
    
    valid_mask = (test_df['label_numeric'].notna()) & (test_df['label_numeric'] >= 0) & (test_df['label_numeric'] < num_labels)
    test_df_valid = test_df[valid_mask].copy()
    
    print(f"Test data after filtering invalid labels: {test_df_valid.shape}")
    if test_df_valid.shape[0] == 0:
        print("ERROR: No valid labels found in test data after conversion!")
        print("First 10 label_id values:", test_df['label_id'].head(10).tolist())
        return
    
    test_df_valid['label_numeric'] = test_df_valid['label_numeric'].astype(int)
    
    test_text_column = 'content'
    
    test_dataset = Dataset.from_dict({
        'text': test_df_valid[test_text_column].tolist()
    })
    
    print("Tokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        ),
        batched=True
    )
    
    print("Running inference...")
    raw_predictions = trainer.predict(tokenized_test_dataset)
    predicted_class_ids = np.argmax(raw_predictions.predictions, axis=1)
    
    test_df_valid['predicted_label'] = predicted_class_ids
    
    print("Calculating metrics and generating reports...")
    
    accuracy = (test_df_valid['predicted_label'] == test_df_valid['label_numeric']).mean()
    print(f"Accuracy on test data: {accuracy:.4f}")
    
    conf_matrix = confusion_matrix(
        test_df_valid['label_numeric'], 
        test_df_valid['predicted_label'],
        labels=range(num_labels)  
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=range(num_labels),
        yticklabels=range(num_labels)
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    print("\nClassification Report:")
    report = classification_report(
        test_df_valid['label_numeric'],
        test_df_valid['predicted_label'],
        labels=range(num_labels),
        digits=4
    )
    print(report)
    

if __name__ == "__main__":
    main()