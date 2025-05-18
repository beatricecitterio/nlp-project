from dataset import TrainDataset, TestDataset
import evaluate
import gc
import json
from huggingface_hub import login
import os
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import wandb


def eval(model, tokenizer, data_loader, limit_loader, max_new_tokens=60, gen_kwargs=None, device="cpu"):

    model.eval()
    gen_kwargs = gen_kwargs or {}

    rouge_func = evaluate.load("rouge")
    best_rouge = 0.0

    bar_eval = tqdm(
        total=limit_loader,
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Evaluation",
        unit="batch"
    )

    for batch, (prompts, true) in enumerate(data_loader):
        prompts = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        # Forward pass
        with torch.inference_mode():
            pred_tokens = model.generate(
                **prompts,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
            )

        # Decode and evaluate
        pred_text = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        pred_text = [t.split('[/INST]')[1] if '[/INST]' in t else t for t in pred_text]
        rouge = rouge_func.compute(
            predictions=pred_text,
            references=true
        )["rougeL"]
        best_rouge += rouge

        bar_eval.set_postfix(
            rouge = "{:.04f}".format(float(best_rouge/(batch+1))),
        )
        bar_eval.update()

        # Empty the cache
        del prompts, pred_tokens, pred_text, true
        torch.cuda.empty_cache()

        if batch >= limit_loader:
            break
    bar_eval.close()
    model.train()

    best_rouge /= limit_loader
    return best_rouge


def train(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, scaler, n_batch_accum, gen_kwargs, num_epochs, save_dir, use_fp16=True, device="cpu"):

    torch.cuda.empty_cache()
    gc.collect()

    # Mount all info abou the run on wandb
    wandb.watch(model, log="all")

    # Track metrics
    best_eval_rouge = -1.0
    checkpoint_id = 0
    
    for epoch in range(num_epochs):
        print("\nEpoch {}/{}".format(epoch+1, num_epochs))

        model.train()
        train_loss = 0.0
        n_opt_steps = 0

        bar_train = tqdm(
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
            position=0,
            desc="Training",
            unit="batch"
        )

        for batch, items in enumerate(train_loader):
            curr_lr = optimizer.param_groups[0]["lr"]

            items = {k: v.to(device, non_blocking=True) for k, v in items.items()}
            with torch.amp.autocast(device, enabled=use_fp16):
                pred_tokens = model(**items)
                loss = pred_tokens.loss

            scaler.scale(loss/n_batch_accum).backward()
            train_loss += loss.item()

            if batch % n_batch_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                n_opt_steps += 1
            
            if (batch+1 == len(train_loader) // 2) or (batch+1 >= len(train_loader)):
                if batch+1 >= len(train_loader):
                    limit_loader = 100
                else:
                    limit_loader = 10
                rouge = eval(model, tokenizer, eval_loader, limit_loader, gen_kwargs=gen_kwargs, device=device)
                if rouge > best_eval_rouge:
                    best_eval_rouge = rouge
                    save_dir_id = os.path.join(save_dir, f"checkpoint-{checkpoint_id}-rouge-{rouge:.4f}")
                    model.save_pretrained(save_dir_id)
                    tokenizer.save_pretrained(save_dir_id)
                    checkpoint_id += 1
                    print("Model saved!")
                
                # Log metrics to wandb
                wandb.log({
                    "train_loss": train_loss/(batch+1),
                    "eval_rouge": best_eval_rouge,
                    "lr": curr_lr,
                    "epoch": epoch,
                })
                
            # Update the learning rate
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            bar_train.set_postfix(
                loss="{:.04f}".format(float(train_loss/(batch+1))),
                lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
                steps="{}".format(n_opt_steps),
            )
            bar_train.update()

            # Empty the cache
            del items, pred_tokens, loss
            torch.cuda.empty_cache()
        bar_train.close()

        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            continue
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()

        # Average the loss over the batches
        train_loss /= len(train_loader)
        print("Train\t Loss: {:.04f}\t Learning rate: {:.04f}".format(train_loss, curr_lr))
        print("Eval\t Rouge: {:.04f}".format(best_eval_rouge))
    
    return best_eval_rouge


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    cwd = os.path.dirname(os.path.abspath(__file__))

    CONFIG = json.load(open(os.path.join(cwd, "config.json")))
    MODEL_ID = CONFIG["model_id"]

    print("Logging in Huggingface...")
    login(token="")

    print("Instantiating the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading the dataset...")
    train_data = TrainDataset(
        data_path=os.path.join(cwd, "data", "prompts_train.jsonl"),
        tokenizer=tokenizer,
        max_len=CONFIG["max_len"],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_data = TestDataset(
        data_path=os.path.join(cwd, "data", "prompts_eval.jsonl")
    )

    test_loader = DataLoader(
        test_data,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    print("Loading and preparing the model for training...")
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
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    print("Instantiating the other training components...")
    optimizer  = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"],
    )

    scaler = torch.GradScaler(enabled=CONFIG["use_fp16"])

    print("Logging in wandb...")
    wandb.login(key="")
    wandb.init(
        name=CONFIG["config_name"],
        project="nlp",
        config=CONFIG
    )

    print("Starting the training phase...")
    gen_kwargs = dict(do_sample=False)

    best_eval_rouge = train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        n_batch_accum=CONFIG["n_batch_accum"],
        gen_kwargs=gen_kwargs,
        num_epochs=CONFIG["epochs"],
        save_dir=os.path.join(cwd, CONFIG["output_dir"]),
        use_fp16=CONFIG["use_fp16"],
        device=DEVICE
    )
    print("Best eval rouge: {:.04f}".format(best_eval_rouge))

    print("Closing wandb...")
    wandb.finish()


if __name__ == "__main__":
    main()