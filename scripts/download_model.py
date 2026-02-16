from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "facebook/nllb-200-distilled-600M"
save_directory = "models/nllb-200-600M"

print(f"Initializing sovereign download of {model_id}...")

# This pulls the weights and the tokenizer files directly
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Save them to your local 'models' folder
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and Tokenizer saved to: {save_directory}")