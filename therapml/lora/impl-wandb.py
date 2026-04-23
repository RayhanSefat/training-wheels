import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 1e-4, 2e-4, 4e-4]
        },
        'lora_r': {
            'values': [8, 16, 32]
        },
        'lora_alpha': {
            'values': [16, 32, 64]
        },
        'batch_size': {
            'values': [4, 8]
        },
        'remove_unused_columns': {
            'values': [True, False]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="rayhansefat-gpt2-lora-tuning")

def train_func():
    with wandb.init() as run:
        config = wandb.config
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("sentence-transformers/eli5", split="train[:1000]")
        val_dataset = load_dataset("sentence-transformers/eli5", split="train[1000:1200]")

        column_names = dataset.column_names 

        def preprocess(examples):
            texts = [q + " " + a for q, a in zip(examples['question'], examples['answer'])]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

        tokenized_train = dataset.map(
            preprocess, 
            batched=True, 
            remove_columns=column_names
        )
        tokenized_val = val_dataset.map(
            preprocess, 
            batched=True, 
            remove_columns=column_names
        )

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)

        training_args = TrainingArguments(
            output_dir="./results",
            report_to="wandb",
            num_train_epochs=2,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=config.remove_unused_columns,
            run_name=run.name 
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        trainer.train()

if __name__ == "__main__":
    wandb.agent(sweep_id, function=train_func, count=20)