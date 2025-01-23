from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

local_model_path = "cache1/model/ahmedrachid/FinancialBERT-Sentiment-Analysis/"

app = Flask(__name__)

print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(local_model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("Model loaded successfully.")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Please provide the 'text' field in the request body."}), 400

    text = data["text"]

    results = nlp([text])
    sentiment = results[0]["label"]
    confidence = results[0]["score"]

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
