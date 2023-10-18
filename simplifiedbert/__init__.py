import gc

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from typing import List

from pathlib import Path

from accelerate import Accelerator

from torch.utils.data import Dataset
from torch.optim import AdamW

from transformers import (BertModel, Trainer, TrainingArguments,
                          EarlyStoppingCallback, AutoTokenizer,
                          AutoModelForSequenceClassification)

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split


acc = Accelerator()


class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class SimplifiedBert(nn.Module):
    
    def __init__(self, model_name, classes:int = 2, num_labels:int = 2, epochs:int = 20,
                 batch_size:int = 8, batch_status:int = 32, learning_rate=1e-5,
                 early_stop:int = 3, max_tokens:int = 200, eval_steps:int = 100, max_steps=-1,
                 output_dir: Path = 'model_outputs', logging_dir: Path = None, device: str = None,
                 *args, **kwargs):

        super(SimplifiedBert, self).__init__(*args, **kwargs)

        self.model_name = model_name
        self.classes = classes
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_status = batch_status
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.max_tokens = max_tokens
        self.eval_steps = eval_steps
        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)


    @classmethod
    def clear_env(cls):
        torch.cuda.empty_cache()
        gc.collect()


    def evaluate(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy, "f1": f1}


    def preprocess_function(self, texts):
        return self.tokenizer(
            texts, padding=True, return_tensors='pt',
            truncation=True, max_length=self.max_tokens,
        )

    def cross_validation(self, dataset_features, dataset_labels, k: int = 10, test_size: int = 0.1) -> List[dict]:
        data = []
        for k in range(k):
            data_train, data_test, y_train, y_test = train_test_split(
                dataset_features, dataset_labels, test_size=test_size, random_state=k
            )
            
            df_train = pd.concat([data_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1, ignore_index=True)
            df_test = pd.concat([data_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1, ignore_index=True)
    
            metrics, trainer = self.fit(
                df_train[0].values.tolist(), df_train[1].values.tolist(),
                df_test[0].values.tolist(), df_test[1].values.tolist()
            )
            data.append({'metrics': metrics, 'k': k})
        return data

    def fit(self, train_dataset, train_label, eval_dataset, eval_label):
        args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='steps',
            eval_steps=self.eval_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            seed=1,
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            max_steps=self.max_steps,
            save_steps=self.eval_steps,
            overwrite_output_dir=True
        )

        train_encodings = self.preprocess_function(train_dataset)
        eval_encodings = self.preprocess_function(eval_dataset)

        train_dataset = CustomDataset(train_encodings, train_label)
        eval_dataset = CustomDataset(eval_encodings, eval_label)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.evaluate,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stop)],
        )

        trainer.train()
        metrics = trainer.evaluate()

        return metrics, trainer
