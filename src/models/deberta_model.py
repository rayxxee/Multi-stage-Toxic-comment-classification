import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class DeBERTaClassifier:
    def __init__(self, model_name="microsoft/deberta-v3-small", max_length=256, batch_size=16, num_epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_dataset(self, texts, labels=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if labels is not None:
            dataset_dict['labels'] = labels
            
        return Dataset.from_dict(dataset_dict)

    def train(self, texts, labels, output_dir="./models/deberta_checkpoints"):
        print("Training DeBERTa model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=3
        ).to(self.device)

        train_dataset = self._prepare_dataset(texts, labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()
        print("DeBERTa trained.")

    def predict(self, texts):
        if self.model is None or self.tokenizer is None:
             raise ValueError("Model or tokenizer not loaded")
             
        dataset = self._prepare_dataset(texts)
        
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        
        logits = predictions.predictions
        # predict the class with highest logit
        labels = np.argmax(logits, axis=1)
        return labels
        
    def predict_proba(self, texts):
        if self.model is None or self.tokenizer is None:
             raise ValueError("Model or tokenizer not loaded")
             
        dataset = self._prepare_dataset(texts)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        
        logits = predictions.predictions
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def save(self, filepath):
        if self.model is not None and self.tokenizer is not None:
            print(f"Saving DeBERTa to {filepath}")
            self.model.save_pretrained(filepath)
            self.tokenizer.save_pretrained(filepath)
        else:
            print("Model not trained, cannot save.")

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        instance.tokenizer = AutoTokenizer.from_pretrained(filepath)
        instance.model.to(instance.device)
        return instance
