from transformers import BertTokenizer, BertForSequenceClassification, pipeline

local_model_path = "cache1/model/ahmedrachid/FinancialBERT-Sentiment-Analysis/"

print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(local_model_path)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("Model loaded successfully. Ready for inference.")

print("\nEnter a sentence for sentiment analysis or type 'exit' to quit.")
while True:
    user_input = input(">> ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    results = nlp([user_input])
    print(f"Sentiment: {results[0]['label']}, Confidence: {results[0]['score']:.4f}")
