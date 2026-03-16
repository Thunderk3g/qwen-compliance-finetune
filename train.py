import os
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported

# -----------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# -----------------------------------------------------------------------------
DATA_DIR = "dataset_2.1_rl"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" # Note: using 2.5 as requested by the 4B/3B spec
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
                {"role": "user", "content": f"{instruction}\n\n{user_input}"},
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
class JsonLoggingCallback(TrainerCallback):
    def __init__(self, log_path="training_metrics.json"):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            json.dump([], f)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            try:
                with open(self.log_path, "r") as f:
                    history = json.load(f)
            except:
                history = []
            history.append(logs)
            with open(self.log_path, "w") as f:
                json.dump(history, f)

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
    callbacks=[JsonLoggingCallback()],
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
