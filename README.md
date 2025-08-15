# Emotion-Aware Keyword Extraction Model

## 📌 Overview
The **Emotion-Aware Keyword Extraction Model** is an NLP project that performs two key tasks:
1. **Emotion Classification** – Predicts the underlying emotion from a given text input.
2. **Keyword Extraction** – Identifies specific words or phrases in the text that contribute most to the predicted emotion.

This dual approach enables **Explainable AI (XAI)** for sentiment analysis, making the model not just predictive but also interpretable — particularly useful for **mental health monitoring, social media analysis, and conversational AI applications**.

---

## 🛠 Features
- Emotion detection using **transformer-based architectures** (DistilBERT).
- Keyword extraction for interpretability.
- Hugging Face `Trainer` API for easy training & evaluation.
- Built-in evaluation metrics: Accuracy, Precision, Recall, F1-score.
- Saves fine-tuned model and tokenizer for deployment.

---

## 📂 Project Structure
```
Emotion-Aware-Keyword-Extraction-Model/
│
├── emotion_model/             # Saved trained model & tokenizer
├── train_dataset.csv          # Training dataset
├── test_dataset.csv           # Testing dataset
├── train.py                   # Training script for emotion classification
├── keyword_extraction.py      # Script for extracting emotion keywords
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Tech Stack
```python
stack: ["Python", "PyTorch", "Transformers (Hugging Face)", "Pandas", "scikit-learn", "Hugging Face Datasets", "SpaCy/NLTK (for keyword extraction)"]
```

---

## 📊 Model Architecture
- **Base Model:** `distilbert-base-uncased`
- **Layers:** Transformer encoder → Classification head
- **Output:** 5 emotion labels (configurable)
- **Keyword Extraction:** NER-based + statistical methods (e.g., TF-IDF) to identify contributing terms.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MirAsimAli/Emotion-Aware-Keyword-Extraction-Model.git
cd Emotion-Aware-Keyword-Extraction-Model
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py
```
This will:
- Load datasets (`train_dataset.csv`, `test_dataset.csv`)
- Tokenize & train the model
- Save the fine-tuned model to `./emotion_model`

### 4️⃣ Extract Keywords
```bash
python keyword_extraction.py
```

---

## 📈 Evaluation Metrics
The model reports:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## 💡 Example Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./emotion_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "I feel so anxious about my exam tomorrow."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_label = torch.argmax(outputs.logits, dim=1).item()

print("Predicted Emotion:", predicted_label)
```

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
