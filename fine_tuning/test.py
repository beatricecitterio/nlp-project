from huggingface_hub import login
import json
import os
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


if __name__ == "__main__":
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    cwd = os.path.dirname(os.path.abspath(__file__))

    CONFIG = json.load(open(os.path.join(cwd, "config.json")))
    MODEL_ID = CONFIG["model_id"]

    print("Logging in Huggingface...")
    login(token="")
    checkpoint_path = os.path.join(cwd, "minerva-350M")

    print("Instantiating the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    model.eval()

    prompt = "<s>[INST] Genera un probabile tweet di un politico con:\nPartito di appartenenza: Destra\nContenuto: Riforma fiscale e conservatorismo economico\nTono: Critico / Negativo [/INST]"
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)
    gen_kwargs = dict(do_sample=True, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2)
    with torch.no_grad():
        pred_tokens = model.generate(
            **prompt_enc,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=80,
            **gen_kwargs
        )
    
    pred_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
    pred_text = pred_text.split('[/INST]')[1] if '[/INST]' in pred_text else pred_text
    print("Generated tweet:")
    print(pred_text)