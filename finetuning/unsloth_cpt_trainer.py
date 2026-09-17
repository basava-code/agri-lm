#!/usr/bin/env python3
"""Production Unsloth CPT Trainer for Agri-LM (v2).

- Full 16-bit LoRA across ALL linear projections
  (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Sequence packing, cosine LR decay with warmup, unsloth gradient checkpointing
- Standalone merged 16-bit checkpoint export (clean base + adapter fusion)
- Optional GGUF export for Ollama serving

Example:
    python unsloth_cpt_trainer.py \
        --model_name unsloth/gemma-4-e2b-it \
        --dataset_path "data/processed/Acid Soils of India/Acid Soils of India_cpt_dataset_packed.jsonl" \
        --output_dir outputs/cpt/gemma4_e2b_acid_soils_v2 \
        --merged_dir outputs/cpt/gemma4_e2b_acid_soils_v2_merged \
        --epochs 2 --rank 64 --max_seq_length 4096
"""
import argparse
import inspect
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import unsloth
except ImportError:
    unsloth = None

from datasets import Dataset, load_dataset


def _text_tokenizer(tok):
    """Unsloth returns Gemma4Processor for multimodal models; the plain text
    tokenizer lives at .tokenizer."""
    inner = getattr(tok, "tokenizer", None)
    return inner if inner is not None else tok


def manually_pack_dataset(ds, tokenizer, max_seq_length: int):
    """True CPT packing: concatenate documents into dense max_seq_length token
    blocks (<bos> doc <eos> doc <eos> ... ). Needed because TRL refuses native
    packing for vision-language models like gemma-4-e2b/e4b."""
    tok = _text_tokenizer(tokenizer)
    eos = tok.eos_token_id or 1

    streams: list[list[int]] = []
    for sample in ds:
        text = str(sample["text"]).strip()
        if text.startswith("<bos>"):
            text = text[5:]
        if text.endswith("<eos>"):
            text = text[:-5]
        ids = tok.encode(text.strip(), add_special_tokens=False)
        if ids:
            streams.append(ids + [eos])

    blocks: list[str] = []
    current: list[int] = []
    dropped = 0
    for ids in streams:
        if len(ids) > max_seq_length - 1:
            ids = ids[:max_seq_length - 1]
            dropped += 1
        if current and len(current) + len(ids) > max_seq_length - 1:
            blocks.append(tok.decode(current))
            current = []
        current.extend(ids)
    if current:
        blocks.append(tok.decode(current))

    total_tokens = sum(len(s) for s in streams)
    print(f"[Trainer] Manual CPT packing: {len(ds)} docs ({total_tokens:,} tokens) -> {len(blocks)} "
          f"dense blocks (max_seq_length={max_seq_length}, {dropped} overlong docs truncated)")
    return Dataset.from_dict({"text": blocks})


def is_vision_language_model(model) -> bool:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    return (
        getattr(cfg, "vision_config", None) is not None
        or hasattr(cfg, "text_config")
        or getattr(cfg, "model_type", "") in ("gemma3", "gemma4", "llava", "paligemma")
    )


def parse_args():
    p = argparse.ArgumentParser(description="Unsloth CPT Fine-Tuning (production)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name", type=str, default="unsloth/gemma-4-e2b-it", help="Base model (gemma-4-e2b-it / gemma-4-e4b-it / gemma-2-2b-it)")
    p.add_argument("--dataset_path", type=str, required=True, help="JSONL CPT dataset with a 'text' field")
    p.add_argument("--output_dir", type=str, default="./outputs/cpt/run", help="Adapter/checkpoint output dir")
    p.add_argument("--merged_dir", type=str, default=None, help="Standalone merged 16-bit model dir (default: <output_dir>_merged)")
    p.add_argument("--export_gguf", type=str, default=None, help="Optional GGUF export path (e.g. model_q8_0.gguf)")
    p.add_argument("--gguf_quant", type=str, default="f16", choices=["q8_0", "q4_k_m", "q5_k_m", "f16"], help="GGUF quantization method")
    p.add_argument("--merge_from_adapter", type=str, default=None, help="Skip training; load this adapter on the base and merge-export it")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--packing", type=str, default="auto", choices=["auto", "manual", "off"],
                   help="auto: manual dense packing for VLMs (TRL forbids native VLM packing), TRL packing otherwise; manual: always manual; off: padded windows")
    p.add_argument("--batch_size", type=int, default=8, help="Per-device batch size (packing makes sequences dense)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=2e-5, help="Low LR for stable CPT")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=64, help="LoRA rank (high capacity for fact injection)")
    p.add_argument("--lora_alpha", type=int, default=None, help="Defaults to 2x rank")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="0.0 is fastest with Unsloth kernels")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--optim", type=str, default="adamw_torch_fused", choices=["adamw_torch_fused", "paged_adamw_8bit", "adamw_8bit"])
    p.add_argument("--smoke_test", action="store_true", help="Run a tiny generation check after training")
    return p.parse_args()


def filter_config_kwargs(config_cls, kwargs: dict) -> dict:
    valid = set(inspect.signature(config_cls.__init__).parameters)
    dropped = {k: v for k, v in kwargs.items() if k not in valid}
    if dropped:
        print(f"[Trainer] Dropping config args unsupported by installed TRL version: {list(dropped)}")
    return {k: v for k, v in kwargs.items() if k in valid}


def load_base_model(args):
    from unsloth import FastLanguageModel
    try:
        from unsloth import FastModel as AltLoader
    except ImportError:
        AltLoader = None

    loader_kwargs = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=None,
        full_finetuning=False,
    )
    try:
        return FastLanguageModel.from_pretrained(**loader_kwargs)
    except TypeError:
        loader_kwargs.pop("full_finetuning", None)
        return FastLanguageModel.from_pretrained(**loader_kwargs)
    except Exception as e:
        if AltLoader is None:
            raise
        print(f"[Trainer] FastLanguageModel failed ({e!r}); retrying with FastModel.")
        try:
            return AltLoader.from_pretrained(**loader_kwargs)
        except TypeError:
            loader_kwargs.pop("full_finetuning", None)
            return AltLoader.from_pretrained(**loader_kwargs)


def build_peft_model(model, args):
    from unsloth import FastLanguageModel
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    peft_kwargs = dict(
        r=args.rank,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha or args.rank * 2,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    try:
        return FastLanguageModel.get_peft_model(model, **peft_kwargs)
    except TypeError as e:
        print(f"[Trainer] PEFT kwargs trimmed for compatibility: {e}")
        for key in ("use_rslora", "loftq_config", "bias"):
            peft_kwargs.pop(key, None)
        return FastLanguageModel.get_peft_model(model, **peft_kwargs)


def validate_dataset(path: str, text_field: str = "text"):
    ds = load_dataset("json", data_files=path, split="train")
    if len(ds) == 0:
        raise ValueError(f"Dataset at {path} is empty.")
    sample = ds[0]
    if text_field not in sample:
        raise ValueError(f"Dataset missing '{text_field}' field. Columns: {list(sample)}")
    preview = str(sample[text_field])[:300].replace("\n", " ")
    print(f"[Trainer] Dataset size: {len(ds):,} samples")
    print(f"[Trainer] First sample preview: {preview}...")
    return ds


def smoke_test(model, tokenizer, prompt: str = "Question: What percentage of cropped land in India has acid soils?\nAnswer:"):
    from transformers import TextStreamer
    tok = _text_tokenizer(tokenizer)
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    print("[Trainer] Smoke test generation:")
    _ = model.generate(**inputs, max_new_tokens=128, temperature=0.3, streamer=TextStreamer(tok, skip_prompt=True))


def export_merged(model, tokenizer, merged_dir: str, gguf_path: str | None, gguf_quant: str):
    model.config.use_cache = True
    tok = _text_tokenizer(tokenizer)
    os.makedirs(merged_dir, exist_ok=True)
    save_merged = getattr(model, "save_pretrained_merged", None)
    if save_merged is not None:
        save_merged(merged_dir, tok, save_method="merged_16bit")
        print(f"[Trainer] Merged 16-bit model saved to {merged_dir}")
    else:
        print("[Trainer] save_pretrained_merged unavailable; manual merge_and_unload fallback.")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[Trainer] Merged model saved to {merged_dir}")

    if gguf_path:
        save_gguf = getattr(model, "save_pretrained_gguf", None)
        if save_gguf is not None:
            save_gguf(gguf_path, tok, quantization_method=gguf_quant)
            print(f"[Trainer] GGUF exported to {gguf_path} ({gguf_quant})")
        else:
            print("[Trainer] GGUF export unsupported by this Unsloth version; skipping.")


def main():
    args = parse_args()
    merged_dir = args.merged_dir or f"{args.output_dir.rstrip('/')}_merged"

    from trl import SFTConfig, SFTTrainer

    if args.merge_from_adapter:
        print(f"[Trainer] Merge-only mode: loading adapter {args.merge_from_adapter}")
        merge_args = argparse.Namespace(**{**vars(args), "model_name": args.merge_from_adapter})
        model, tokenizer = load_base_model(merge_args)
        export_merged(model, tokenizer, merged_dir, args.export_gguf, args.gguf_quant)
        return

    model, tokenizer = load_base_model(args)
    model = build_peft_model(model, args)
    print(f"[Trainer] LoRA rank={args.rank} alpha={args.lora_alpha or args.rank * 2} targets=all 7 linear projections (16-bit)")

    ds = validate_dataset(args.dataset_path)

    vlm = is_vision_language_model(model)
    use_manual_packing = args.packing == "manual" or (args.packing == "auto" and vlm)
    use_trl_packing = args.packing != "off" and not use_manual_packing
    if args.packing == "auto" and vlm:
        print("[Trainer] Vision-language base detected: TRL forbids native VLM packing -> using manual dense CPT packing.")
    if use_manual_packing:
        ds = manually_pack_dataset(ds, tokenizer, args.max_seq_length)

    num_blocks = len(ds)
    batch_size, grad_accum = args.batch_size, args.gradient_accumulation_steps
    min_steps = 16

    def _steps(bs: int, ga: int) -> int:
        import math
        return max(1, math.ceil(num_blocks / max(1, bs * ga))) * int(math.ceil(args.epochs))

    while _steps(batch_size, grad_accum) < min_steps and (batch_size > 1 or grad_accum > 1):
        if batch_size > 1:
            batch_size //= 2
        else:
            grad_accum //= 2
    total_steps = _steps(batch_size, grad_accum)
    if (batch_size, grad_accum) != (args.batch_size, args.gradient_accumulation_steps):
        print(f"[Trainer] Tiny-corpus auto-tuning: batch_size {args.batch_size}->{batch_size}, "
              f"grad_accum {args.gradient_accumulation_steps}->{grad_accum} "
              f"(total optimizer steps: {total_steps})")
    if total_steps < 16:
        print(f"[Trainer] WARNING: only {total_steps} optimizer steps. For real knowledge injection "
              f"raise --epochs to 4-6 or add more books to the corpus.")

    warmup_steps = min(max(5, int(args.warmup_ratio * total_steps)), max(1, total_steps // 10))
    logging_steps = min(args.logging_steps, max(1, total_steps // 5))
    print(f"[Trainer] {num_blocks} blocks | ~{total_steps} optimizer steps | warmup_steps={warmup_steps}")

    config_kwargs = {
        "output_dir": args.output_dir,
        "dataset_text_field": "text",
        "packing": use_trl_packing,
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,
        "bf16": True,
        "optim": args.optim,
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_num_workers": 2,
        "dataset_num_proc": 8,
    }
    train_kwargs = filter_config_kwargs(SFTConfig, config_kwargs)
    trainer_kwargs = {"model": model, "train_dataset": ds, "args": SFTConfig(**train_kwargs)}
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    print("[Trainer] Starting CPT training...")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"[Trainer] Final adapter saved to {final_adapter_dir}")

    if args.smoke_test:
        smoke_test(model, tokenizer)

    export_merged(model, tokenizer, merged_dir, args.export_gguf, args.gguf_quant)
    print("[Trainer] Done.")


if __name__ == "__main__":
    main()
