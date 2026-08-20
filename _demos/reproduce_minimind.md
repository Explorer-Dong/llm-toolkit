# Minimind 复现手册

> 从零开始开发一个 LLM，包括预训练、SFT、RL 等。

## Data

```bash
# hf
hf download jingyaogong/minimind_dataset --repo-type=dataset --local-dir ./dataset/ --include "*.jsonl"

# ms
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset/ --include "*.jsonl"
```

## Train

### Pretrain

```bash
# pretrain
cd trainer
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 train_pretrain.py \
  --max_seq_len 380 \
  --batch_size 128 \
  --data_path ../dataset/pretrain_t2t.jsonl \
  --use_moe 1 \
  --use_wandb \
  --wandb_project llm-mini
```

### Full SFT

```bash
cd trainer
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 train_full_sft.py \
  --max_seq_len 768 \
  --batch_size 128 \
  --data_path ../dataset/sft_t2t.jsonl \
  --use_moe 1 \
  --use_wandb \
  --wandb_project llm-mini
```

### (Optional) Distillation

```bash
cd trainer
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 train_distillation.py \
  --use_wandb \
  --wandb_project llm-mini
```

### (Optional) LoRA SFT

```bash
cd trainer
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 train_lora.py \
  --use_wandb \
  --wandb_project llm-mini

cd trainer
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 4 train_lora.py \
  --use_wandb \
  --wandb_project llm-mini \
  --lora_name lora_exam \
  --data_path ../dataset/lora_exam.jsonl
```

### RL

## Evaluation

```bash
python eval_llm.py --weight pretrain
```
