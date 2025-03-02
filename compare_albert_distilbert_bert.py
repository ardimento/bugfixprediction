import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Model names
models = {
    "ALBERT": "albert-base-v2",
    "DistilBERT": "distilbert-base-uncased",
    "Google BERT": "bert-base-uncased"
}

# Load dataset from JSON file (handling JSON lines format)
def load_dataset(json_file):
    bug_reports = []
    labels = []
    
    with open(json_file, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                bug = entry["bug"]
                title = bug["short_desc"]
                comments = " ".join([desc["thetext"] for desc in bug.get("long_desc", [])])
                full_text = title + " " + comments
                bug_reports.append(full_text)
                labels.append(int(bug["days_resolution"]))
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
    
    return bug_reports, labels

# Preprocessing function
def preprocess_data(tokenizer, texts, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

# Function to evaluate models
def evaluate_model(model_name, model, tokenizer, texts, labels):
    model.eval()
    inputs = preprocess_data(tokenizer, texts)
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        inference_time = (time.time() - start_time) / len(texts) * 1000  # Convert to ms/sample
    
    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    labels = np.array(labels)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    
    return acc, f1, rmse, inference_time, predictions

# Load dataset
json_file = "LiveCode_original.json"
bug_reports, labels = load_dataset(json_file)

# Iterate over models and evaluate
results = {}
predictions_data = []

for model_name, model_path in models.items():
    print(f"Evaluating {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)  # Regression task
    
    acc, f1, rmse, inference_time, predictions = evaluate_model(model_name, model, tokenizer, bug_reports, labels)
    results[model_name] = {
        "Accuracy": acc,
        "F1-score": f1,
        "RMSE": rmse,
        "Inference Time (ms/sample)": inference_time
    }
    
    for i, pred in enumerate(predictions):
        predictions_data.append([model_name, bug_reports[i], labels[i], pred])

# Save results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv", index=True)

predictions_df = pd.DataFrame(predictions_data, columns=["Model", "Bug Report", "Actual Fix Time", "Predicted Fix Time"])
predictions_df.to_csv("predictions.csv", index=False)

# Generate and save plots
metrics = ["Accuracy", "F1-score", "RMSE", "Inference Time (ms/sample)"]
for metric in metrics:
    plt.figure(figsize=(6, 4))
    values = [results[model][metric] for model in models.keys()]
    plt.bar(models.keys(), values, color=['blue', 'green', 'red'])
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} among ALBERT, DistilBERT, and Google BERT")
    plt.xticks(rotation=45)
    plt.savefig(f"{metric.replace(' ', '_').lower()}.png")
    plt.close()

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
