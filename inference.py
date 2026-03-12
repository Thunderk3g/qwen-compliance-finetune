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
insurance_article = """
If you crash your car, we will pay you money. You need to call us within 2 days or we won't pay. 
Also, don't lie to us because if we find out, we keep the money.
"""

# Format input to match the structure we trained on
messages = [
    {"role": "system", "content": "You are a compliance-focused insurance documentation assistant."},
    {"role": "user", "content": f"Rewrite the content to meet insurance regulatory and compliance standards.\n\n{insurance_article}"}
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
