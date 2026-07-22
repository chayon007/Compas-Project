"""Baseline model training."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from typing import Callable


def train_tfidf_baseline(X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        max_features: int = 30000,
                        ngram_range: Tuple[int, int] = (1, 2)) -> Dict:
    """
    Train TF-IDF + Logistic Regression baseline.
    
    Args:
        X_train: Training texts
        X_test: Test texts
        y_train: Training labels
        y_test: Test labels
        max_features: Max TF-IDF features
        ngram_range: N-gram range
    
    Returns:
        Dictionary with model, vectorizer, and metrics
    """
    print("Training TF-IDF + Logistic Regression baseline...")
    
    # Vectorize text
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)
    
    # Train LR classifier
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    clf.fit(X_train_t, y_train)
    
    # Predictions
    y_pred_train = clf.predict(X_train_t)
    y_pred_test = clf.predict(X_test_t)
    
    # Metrics
    print("\n=== Training Set ===")
    train_report = classification_report(y_train, y_pred_train, digits=4, zero_division=0)
    print(train_report)
    
    print("\n=== Test Set ===")
    test_report = classification_report(y_test, y_pred_test, digits=4, zero_division=0)
    print(test_report)
    
    return {
        'model': clf,
        'vectorizer': vec,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_report': train_report,
        'test_report': test_report,
        'test_accuracy': (y_pred_test == y_test).mean(),
    }


def train_transformer_baseline(
    train_texts: list,
    train_labels: list,
    test_texts: list,
    test_labels: list,
    model_name: str = "google-bert/bert-base-multilingual-uncased",
    output_dir: Path = Path("models/transformer_baseline"),
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = "cuda",
    save_model: bool = True
) -> Dict:
    """
    Train transformer baseline (BanglaBERT or XLM-R).
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        model_name: HuggingFace model name
        output_dir: Where to save models
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cuda or cpu)
        save_model: Whether to save model
    
    Returns:
        Dictionary with trained model and metrics
    """
    print(f"Training {model_name} baseline...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        hidden_dropout_prob=0.1
    )
    
    # Tokenize data
    def tokenize_function(texts):
        return tokenizer(texts, truncation=True, max_length=512, padding=True)
    
    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)
    
    # Create Dataset objects
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels
    })
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=42,
        device=device
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    
    # Save
    if save_model:
        model.save_pretrained(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        print(f"Model saved to {output_dir / 'final_model'}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'trainer': trainer,
        'eval_results': eval_results,
        'output_dir': output_dir
    }
