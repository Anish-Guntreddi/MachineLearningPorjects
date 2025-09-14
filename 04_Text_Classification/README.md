# Text Classification Project - IMDb Sentiment/AG News

## 1. Problem Definition & Use Case

**Problem:** Automatically categorize text documents into predefined classes based on content.

**Use Case:** Text classification enables:
- Sentiment analysis for customer feedback
- Email spam detection
- News categorization
- Support ticket routing
- Content moderation

**Business Impact:** Reduces manual content review by 85%, enables real-time customer sentiment monitoring, and improves response times by 60%.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets

1. **IMDb Movie Reviews**
```python
from datasets import load_dataset
imdb = load_dataset('imdb')
# 25,000 training, 25,000 test samples
# Binary classification: positive/negative
```

2. **AG News**
```python
from torchtext.datasets import AG_NEWS
train_iter, test_iter = AG_NEWS()
# 120,000 training, 7,600 test samples
# 4 classes: World, Sports, Business, Sci/Tech
```

### Data Schema
```python
{
    'text': str,           # Raw text content
    'label': int,          # Class label
    'tokens': List[str],   # Tokenized text
    'embeddings': tensor,  # Text embeddings
    'metadata': dict       # Additional info
}
```

### Text Preprocessing Pipeline
```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Lowercase
        text = text.lower()
        return text
    
    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct]
        return tokens
```

### Advanced Preprocessing
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def prepare_features(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
```

## 3. Baseline Models

### TF-IDF + Logistic Regression
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

baseline_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,3))),
    ('clf', LogisticRegression(max_iter=1000))
])
```
**Expected Performance:** 88-90% accuracy on IMDb

### FastText Classifier
```python
import fasttext

# Prepare data in FastText format
def prepare_fasttext_data(texts, labels, filename):
    with open(filename, 'w') as f:
        for text, label in zip(texts, labels):
            f.write(f'__label__{label} {text}\n')

model = fasttext.train_supervised(
    'train.txt',
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    dim=100
)
```
**Expected Performance:** 91-92% accuracy

## 4. Advanced/Stretch Models

### 1. BERT Fine-tuning
```python
from transformers import BertForSequenceClassification, Trainer

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
```

### 2. RoBERTa with Custom Head
```python
from transformers import RobertaModel
import torch.nn as nn

class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))
```

### 3. Ensemble Methods
```python
class TextEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, texts):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(texts)
            predictions.append(pred)
        
        # Weighted average
        weights = [0.3, 0.4, 0.3]  # BERT, RoBERTa, XLNet
        final_pred = np.average(predictions, weights=weights, axis=0)
        return final_pred.argmax(axis=1)
```

**Target Performance:** 95%+ accuracy on IMDb

## 5. Training Details

### Data Loading
```python
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### Training Configuration
```python
config = {
    'learning_rate': 2e-5,
    'batch_size': 32,
    'epochs': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 2,
    'fp16': True,
    'max_grad_norm': 1.0,
    'scheduler': 'cosine',
    'label_smoothing': 0.1
}
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb

wandb.init(project='text-classification', config=config)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_metrics = evaluate(model, val_loader)
    
    wandb.log({
        'train_loss': train_loss,
        'val_accuracy': val_metrics['accuracy'],
        'val_f1': val_metrics['f1']
    })
    
    # Log confusion matrix
    wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
        probs=None,
        y_true=val_labels,
        preds=val_preds,
        class_names=class_names
    )})
```

## 8. Deployment Pathway

### FastAPI Service
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    # Preprocess
    inputs = tokenizer(request.text, return_tensors='pt')
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    predicted_class = probs.argmax().item()
    confidence = probs.max().item()
    
    return PredictionResponse(
        label=class_names[predicted_class],
        confidence=confidence,
        probabilities={name: float(p) for name, p in 
                      zip(class_names, probs[0])}
    )
```

### Gradio Interface
```python
import gradio as gr

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    return {class_names[i]: float(probs[i]) 
            for i in range(len(class_names))}

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=5, label="Enter text"),
    outputs=gr.Label(num_top_classes=3),
    examples=[
        "This movie was absolutely fantastic!",
        "Terrible waste of time.",
        "The latest smartphone features AI capabilities."
    ],
    title="Text Classification Demo"
)

demo.launch()
```

## 9. Extensions & Research Directions

### Advanced Techniques
1. **Multi-task Learning**
2. **Few-shot Classification**
3. **Active Learning**
4. **Adversarial Training**

### Novel Experiments
- Cross-lingual classification
- Zero-shot classification
- Hierarchical classification
- Aspect-based sentiment analysis

## 10. Portfolio Polish

### Project Structure
```
text_classification/
├── README.md
├── configs/
│   └── training_config.yaml
├── data/
│   └── prepare_data.py
├── models/
│   ├── baseline.py
│   └── transformer_models.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── inference.py
├── notebooks/
│   └── exploratory_analysis.ipynb
└── deployment/
    ├── api.py
    └── gradio_app.py
```

### Performance Benchmarks
| Model | IMDb | AG News | Inference (ms) |
|-------|------|---------|----------------|
| TF-IDF + LR | 88.5% | 91.2% | 0.5 |
| FastText | 91.3% | 92.8% | 1.2 |
| BERT | 95.2% | 94.6% | 15.3 |
| RoBERTa | 95.8% | 95.1% | 16.1 |