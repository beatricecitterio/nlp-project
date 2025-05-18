from dataset import TestDataset
from huggingface_hub import login
import json
import os
import pandas as pd
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def generate_tweets(model, tokenizer, data_loader, device, max_new_tokens=64, gen_kwargs=None):

    torch.cuda.empty_cache()
    model.eval()
    
    prompts = []
    tweets = []
    generated_tweets = []
    
    bar_eval = tqdm(
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Generating tweets",
        unit="batch"
    )

    for _, (prompts_batch, tweets_batch) in enumerate(data_loader):
        prompts_batch_enc = tokenizer(prompts_batch, padding=True, return_tensors="pt").to(device)

        # Forward pass
        with torch.inference_mode():
            pred_tokens = model.generate(
                **prompts_batch_enc,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
            )

        # Decode
        pred_text = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        pred_text = [t.split('[/INST]')[1] if '[/INST]' in t else t for t in pred_text]
        bar_eval.update(1)

        prompts.extend(prompts_batch)
        tweets.extend(tweets_batch)
        generated_tweets.extend(pred_text)

        # Empty the cache
        del prompts_batch, pred_tokens
        torch.cuda.empty_cache()
    bar_eval.close()

    return prompts, tweets, generated_tweets


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    cwd = os.path.dirname(os.path.abspath(__file__))

    CONFIG = json.load(open(os.path.join(cwd, "config.json")))
    MODEL_ID = CONFIG["model_id"]

    print("Logging in Huggingface...")
    login(token="")
    checkpoint_path = os.path.join(cwd, "minerva-350M/epoch3")

    print("Instantiating the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading the dataset...")
    test_data = TestDataset(
        data_path=os.path.join(cwd, "data", "prompts_test.jsonl")
    )
    test_loader = DataLoader(
        test_data,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    print("Loading the model...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map=DEVICE
    )
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        device_map=DEVICE
    )
    model = model.to(DEVICE)

    # Start generating tweets
    gen_kwargs = dict(do_sample=True, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2)
    prompts, original_tweets, generated_tweets = generate_tweets(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_loader,
        device=DEVICE,
        max_new_tokens=CONFIG["max_len"],
        gen_kwargs=gen_kwargs
    )
    # Save the generated tweets
    df = pd.DataFrame({
        'prompt': prompts,
        'original_tweet': original_tweets,
        'generated_tweet': generated_tweets
    })
    df.to_csv(os.path.join(cwd, "data", "generated_350m_baseline.csv"), index=False)


if __name__ == "__main__":
    main()