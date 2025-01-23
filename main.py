import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name="ahmedrachid/FinancialBERT-Sentiment-Analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"cache1/tokenizer/{model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(f"cache1/model/{model_name}")