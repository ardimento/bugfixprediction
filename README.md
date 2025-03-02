## **Bug Fixing Time Prediction using ALBERT, DistilBERT, and Google BERT**
This repository contains the implementation for predicting bug-fixing time using transformer-based language models. The models evaluated include **ALBERT, DistilBERT, and Google BERT**. The dataset consists of bug reports extracted from **LiveCode's Bugzilla**, with textual features such as bug descriptions and developer comments used for training.

### **Features**
- 📜 **Data Processing:** Extracts and preprocesses bug reports from a JSON dataset.
- 🔍 **Model Comparison:** Evaluates ALBERT, DistilBERT, and Google BERT for bug-fixing time prediction.
- 📊 **Performance Metrics:** Computes Accuracy, F1-score, RMSE, and Inference Time.
- 📈 **Visualization:** Generates comparative plots for model performance analysis.

### **Installation**
To run this project in **Google Colab**, ensure the required dependencies are installed:
```bash
pip install torch transformers scikit-learn matplotlib
```

### **Usage**
Upload your dataset (`LiveCode_original.json`) and run:
```python
python bug_fixing_time_prediction.py
```

### **Dataset**
- The dataset is extracted from **LiveCode's Bugzilla**.
- Each bug report includes the **short description, developer comments, and resolution time (days_resolution)**.

### **Results**
The models are evaluated based on:
- **Accuracy**: Correctness of predictions.
- **F1-score**: Balance between precision and recall.
- **RMSE**: Measures deviation between predicted and actual bug-fixing time.
- **Inference Time**: Measures model efficiency.
