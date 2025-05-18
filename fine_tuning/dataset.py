from torch.utils.data import Dataset
import json
import torch
    

class TrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):

        self.items = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = f"<s>[INST] {obj['prompt'].strip()} [/INST]"
                tweet = f"{obj['tweet'].strip()} </s>"
                full = prompt+tweet

                enc = tokenizer(
                    full,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length"
                )
                ids = enc["input_ids"]

                # Find the index of the closing [/INST] token
                try:
                    boundary = ids.index(tokenizer.convert_tokens_to_ids("[/INST]")) + 1
                except ValueError:
                    boundary = len(prompt.split())

                # Mask the prompt tokens with -100
                labels = [-100]*boundary + [
                    t if t != tokenizer.pad_token_id else -100
                    for t in ids[boundary:]
                ]
                
                item = {
                    "input_ids" : torch.tensor(ids),
                    "attention_mask" : torch.tensor(enc["attention_mask"]),
                    "labels" : torch.tensor(labels)
                }
                self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
    

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.prompts = []
        self.tweets = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = f"<s>[INST] {obj['prompt'].strip()} [/INST]"
                self.prompts.append(prompt)

                tweet = f"{obj['tweet'].strip()} </s>"
                self.tweets.append(tweet)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, i):
        return self.prompts[i], self.tweets[i]