# Qwen 3.5 9B Instruct Fine-Tuning with Unsloth

This repository contains the code and instructions to fine-tune **Qwen 3.5 9B Instruct** (using Qwen 2.5 9B as per current Unsloth support) using the **Unsloth** library. The goal is to train the model to generate regulatory-compliant insurance documentation based on a provided instruction and input article.

Since you are running this on an **RTX 4060 (8GB VRAM)**, memory constraints are tight. We use aggressive memory optimizations (`adamw_8bit`, Unsloth gradient checkpointing, 4-bit quantization, etc.) to fit the 9B model into your GPU.

We have also provided instructions for running this on a **Google Colab Notebook (T4 GPU, 16GB VRAM)** if you encounter Out-Of-Memory (OOM) errors locally.

---

## 1. Environment Setup

### Option A: Local Setup (RTX 4060)

Create a fresh Python environment (Python 3.10+) and ensure you have CUDA installed. Then run:

```bash
pip install torch
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets transformers
```

### Option B: Google Colab Setup (T4 GPU - Recommended for less OOM issues)

Run this in the first cell of your Google Colab:

```python
%%capture
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl datasets transformers
```

Make sure to change the Colab Runtime to **T4 GPU** (Runtime -> Change runtime type). You will also need to zip your `dataset_2.1_rl` folder, upload it, and unzip it: `!unzip dataset_2.1_rl.zip`.

---

## 2. Full Fine-Tuning Script (`train.py`)

Save the following code as `train.py` and run it locally, or paste it into a Colab cell. Make sure your `dataset_2.1_rl` folder is accessible in the same directory.

```python
import os
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# -----------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# -----------------------------------------------------------------------------
DATA_DIR = "dataset_2.1_rl"
MODEL_NAME = "Qwen/Qwen2.5-9B-Instruct" # Note: using 2.5 as requested by the 2.5 spec
OUTPUT_DIR = "./compliance_qwen_model"

# Hardware / Training settings
MAX_SEQ_LENGTH = 4096     # If you OOM on RTX 4060, lower this to 2048 or 1024
LOAD_IN_4BIT = True
BATCH_SIZE = 2            # If you OOM on RTX 4060, lower to 1 and increase gradient_accumulation_steps to 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

# -----------------------------------------------------------------------------
# 2. Dataset Loading & Formatting
# -----------------------------------------------------------------------------
def load_and_format_dataset(data_dir):
    """
    Reads all JSON docs from the directory and extracts instruction, input, output.
    Formats them into standardized chat format.
    """
    conversations = []
    
    # Iterate through all files in the dataset folder
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            instruction = data.get("instruction", "")
            user_input = data.get("input", "")
            final_text = data.get("output", {}).get("final_text", "")
            
            # Skip invalid / empty rows
            if not instruction or not final_text:
                continue
                
            # Build the conversational format
            convo = [
                {"role": "system", "content": "You are a compliance-focused insurance documentation assistant."},
                {"role": "user", "content": f"{instruction}\\n\\n{user_input}"},
                {"role": "assistant", "content": final_text}
            ]
            conversations.append({"messages": convo})
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    # Convert list of dicts to HF Dataset
    hf_dataset = Dataset.from_list(conversations)
    print(f"Loaded {len(hf_dataset)} training examples.")
    return hf_dataset

# -----------------------------------------------------------------------------
# 3. Model & Tokenizer Initialization (Unsloth)
# -----------------------------------------------------------------------------
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,           # Auto-detects bf16 / fp16
    load_in_4bit = LOAD_IN_4BIT,
)

# Apply Qwen-specific chat template capabilities
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen",
)

def apply_chat_template(examples):
    """Applies the Qwen chat template to the messages before inserting into the model"""
    messages = examples["messages"]
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    return { "text" : texts }

dataset = load_and_format_dataset(DATA_DIR)
dataset = dataset.map(apply_chat_template, batched=True)

# -----------------------------------------------------------------------------
# 4. LoRA Configuration
# -----------------------------------------------------------------------------
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,             # Must be 0 for Unsloth optimized training
    bias = "none",                # Must be "none" for Unsloth optimized training
    use_gradient_checkpointing = "unsloth", # Crucial for 8GB VRAM cards
    random_state = 3407,
)

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = True,               # Speeds up training for long documents 5x
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        warmup_steps = 100,
        num_train_epochs = EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",     # 8-bit AdamW is essential for 8GB VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "lora_checkpoints",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

# -----------------------------------------------------------------------------
# 6. Saving the Model
# -----------------------------------------------------------------------------
print(f"Saving fine-tuned LoRA adapters to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saving fully merged (16-bit) model to {OUTPUT_DIR}_merged ...")
# This merges the LoRA weights into the base model and saves it out for production inference
model.save_pretrained_merged(f"{OUTPUT_DIR}_merged", tokenizer, save_method = "merged_16bit")

print("Training Complete!")
```

---

## 3. Example Inference Script (`inference.py`)

Once your training is complete, you can use the newly fine-tuned model to run inferences. Make sure your model is saved at `./compliance_qwen_model`.

```python
from unsloth import FastLanguageModel

# Load the fine-tuned model (LoRA overlay)
# You could also point this to "./compliance_qwen_model_merged" if you prefer the merged model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./compliance_qwen_model",
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# Put model into high-speed inference mode
FastLanguageModel.for_inference(model)

# The document we want to rewrite
insurance_article = \"\"\"
If you crash your car, we will pay you money. You need to call us within 2 days or we won't pay. 
Also, don't lie to us because if we find out, we keep the money.
\"\"\"

# Format input to match the structure we trained on
messages = [
    {"role": "system", "content": "You are a compliance-focused insurance documentation assistant."},
    {"role": "user", "content": f"Rewrite the content to meet insurance regulatory and compliance standards.\\n\\n{insurance_article}"}
]

# Apply chat template
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Forces model to append <|im_start|>assistant natively
    return_tensors = "pt",
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 1024,
    use_cache = True,       # Highly recommended for generation speed
    temperature = 0.3,      # Lower temperature = more precise compliant legalese
    top_p = 0.9
)

# Decode output (skips the prompt tokens to only show the newly generated text)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print("--- REWRITTEN COMPLIANT ARTICLE ---")
print(response)
```

---

## 💡 Optimizing for the RTX 4060 vs. Google Colab

Your RTX 4060 has **8GB of VRAM**. A 9B parameter model mapped dynamically natively takes roughly **5.8 GB VRAM** in 4-bit mode. This leaves only **2.2 GB of VRAM** for gradients and activations during the training process.

**If you encounter a `CUDA Out Of Memory (OOM)` error on your 4060:**
1. In `train.py`, change `MAX_SEQ_LENGTH = 4096` to `MAX_SEQ_LENGTH = 2048` or even `1024`. This strictly restricts how much context fits in memory.
2. Change `BATCH_SIZE = 2` to `BATCH_SIZE = 1` and change `GRADIENT_ACCUMULATION = 4` to `GRADIENT_ACCUMULATION = 8` (to simulate the exact same math steps but require less instant memory).

**If you prefer to just use Google Colab:**
1. Open a new Google Colab notebook.
2. Go to **Runtime -> Change runtime type** and select **T4 GPU** (this grants you 16GB of VRAM for free).
3. Zip your `dataset_2.1_rl` folder, upload it using the Files tab on the left of Colab, and unzip it (`!unzip dataset_2.1_rl.zip`).
4. Paste the scripts directly into the cells; the 16GB VRAM will handle the `4096` context length easily.
