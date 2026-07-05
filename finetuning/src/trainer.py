import logging
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def get_lora_config(config: dict) -> LoraConfig:
    lora_cfg = config['lora']
    logging.info("Configuring LoRA (PEFT)...")
    return LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type'],
    )

def get_sft_config(config: dict, supports_bf16: bool) -> SFTConfig:
    train_cfg = config['training']
    paths_cfg = config['paths']
    dataset_cfg = config['dataset']
    
    logging.info("Setting up SFT Configuration...")
    return SFTConfig(
        output_dir=paths_cfg['results_dir'],
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        optim=train_cfg['optim'],
        logging_steps=train_cfg['logging_steps'],
        learning_rate=train_cfg['learning_rate'],
        fp16=not supports_bf16,
        bf16=supports_bf16,
        max_grad_norm=train_cfg['max_grad_norm'],
        warmup_steps=train_cfg['warmup_steps'],
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        save_strategy=train_cfg['save_strategy'],
        report_to=train_cfg['report_to'],
        dataset_text_field="text",
        max_length=dataset_cfg['max_length'],
        packing=dataset_cfg['packing'],
    )

def run_training(model, tokenizer, dataset, peft_config, sft_config, adapter_output_dir):
    logging.info("Starting Training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    trainer.train()
    
    logging.info(f"Training finished! Saving LoRA adapter to {adapter_output_dir}...")
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    
    return trainer
