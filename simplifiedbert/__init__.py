"""
Custom BERT-Based Text Classification Module

This module defines a custom BERT-based text classification model and related
utility classes and functions.
It includes a custom dataset class, a simplified BERT model, and functions for
data preprocessing, cross-validation, and model training and evaluation.

Classes:
- CustomDataset: A PyTorch dataset for custom text classification data.
- SimplifiedBert: A custom BERT-based text classification model.

Functions:
- clear_env: Clear the GPU environment by emptying the cache and collecting garbage.
- evaluate: Evaluate model predictions and calculate accuracy and F1 score.
- preprocess_function: Preprocess input text for the model.
- cross_validation: Perform k-fold cross-validation for the model.
- fit: Fit the model to the training data and evaluate on the validation data.

This code is intended for text classification tasks using BERT-based models.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

import torch

from accelerate import Accelerator

from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset

from transformers import (Trainer, TrainingArguments,
                          EarlyStoppingCallback, AutoTokenizer,
                          AutoModelForSequenceClassification)

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

acc = Accelerator()


class CustomDataset(Dataset):
    """
    PyTorch dataset for custom data.

    Args:
        encodings (dict): Input encodings (e.g., tokenized text).
        labels (list, optional): List of labels for the data.
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single data item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing input encodings and labels (if available).
        """
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.encodings['input_ids'])


class SimplifiedBert(nn.Module):
    # pylint: disable=W0223

    """
    Custom BERT-based model for sequence classification.

    Args:
        model_name (str): Name of the pretrained BERT model.
        classes (int): Number of classes for classification.
        num_labels (int): Number of labels for the model.
        device (str, optional): Device to run the model on
            (default is 'cuda' if available, else 'cpu').
    """

    def __init__(self, model_name, classes: int = 2, num_labels: int = 2, device: str = None):
        super().__init__()

        self.model_name = model_name
        self.classes = classes
        self.num_labels = num_labels

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.optimizers = [AdamW(self.model.parameters(), lr=self.learning_rate), ]

    @classmethod
    def clear_env(cls):
        """
        Clear the GPU environment by emptying the cache and collecting garbage.
        """
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self, p):
        """
        Evaluate model predictions and calculate accuracy and F1 score.

        Args:
            p (tuple): A tuple containing predicted probabilities and true labels.

        Returns:
            dict: Dictionary containing accuracy and F1 score.
        """
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy, "f1": f1}

    def preprocess_function(self, texts, max_token=200):
        """
        Preprocess input text for the model.

        Args:
            texts (list): List of input texts to preprocess.
            max_token (int): Maximum number of tokens for padding/truncation.

        Returns:
            dict: Tokenized and preprocessed inputs.
        """
        return self.tokenizer(
            texts, padding=True, return_tensors='pt',
            truncation=True, max_length=max_token,
        )

    def cross_validation(
            self, dataset_features, dataset_labels, args: TrainingArguments,
            k: int = 10, test_size: int = 0.1, **kwargs
    ) -> List[dict]:
        """
        Perform k-fold cross-validation for the model.

        Args:
            dataset_features (list): List of features.
            dataset_labels (list): List of labels.
            args (TrainingArguments): Training arguments for the model.
            k (int, optional): Number of cross-validation folds (default is 10).
            test_size (int, optional): Size of the test set (default is 0.1).

        Returns:
            list: List of dictionaries containing metrics and fold information.

        This function performs k-fold cross-validation on the model using the provided dataset.
        It splits the data into training and validation sets for each fold, trains the model,
        and collects evaluation metrics for each fold.

        Example:
            >>> model = SimplifiedBert("bert-base-uncased", classes=2)
            >>> args = TrainingArguments(...)
            >>> dataset_features = ...
            >>> dataset_labels = ...
            >>> results = model.cross_validation(dataset_features, dataset_labels, args, k=5)
        """

        data = []
        for n in range(k):
            data_train, data_test, y_train, y_test = train_test_split(
                dataset_features, dataset_labels, test_size=test_size, random_state=n
            )

            df_train = pd.concat([
                data_train.reset_index(drop=True), y_train.reset_index(drop=True)
            ], axis=1, ignore_index=True)
            df_test = pd.concat([
                data_test.reset_index(drop=True), y_test.reset_index(drop=True)
            ], axis=1, ignore_index=True)

            metrics, _ = self.fit(
                (df_train[0].values.tolist(), df_train[1].values.tolist()),
                (df_test[0].values.tolist(), df_test[1].values.tolist()),
                args, **kwargs
            )
            data.append({'k': n, 'metrics': metrics, })
        return data

    def fit(self, train_dataset: Tuple[List, List], eval_dataset: Tuple[List, List],
            args: TrainingArguments, early_stop=3, learning_rate=5e-5, **kwargs):
        """
        Fit the model to the training data and evaluate on the validation data.

        Args:
            train_dataset (Tuple[List, List]): A tuple containing training data and labels.
            eval_dataset (Tuple[List, List]): A tuple containing evaluation data and labels.
            args (TrainingArguments): Training arguments for the model.
            early_stop (int, optional): Number of patience epochs for early stopping (default is 3).
            learning_rate (float, optional): Learning rate for optimization (default is 5e-5).
            **kwargs: Additional keyword arguments to pass to the Trainer constructor.

        Returns:
            tuple: A tuple containing evaluation metrics and the Trainer instance.

        The function preprocesses the input data, creates custom datasets, and trains the model.

        Additional keyword arguments can be passed to the Trainer constructor to
            further customize the training process.

        Example:
        >>> model = SimplifiedBert("bert-base-uncased", classes=2)
        >>> args = TrainingArguments(...)
        >>> train_data = ...
        >>> eval_data = ...
        >>> results = model.fit(train_data, eval_data, args, early_stop=4, learning_rate=3e-5, weight_decay=0.01)
        """
        train_encodings = self.preprocess_function(train_dataset[0])
        eval_encodings = self.preprocess_function(train_dataset[1])

        train_dataset = CustomDataset(train_encodings, eval_dataset[0])
        eval_dataset = CustomDataset(eval_encodings, eval_dataset[1])

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            learning_rate=learning_rate,
            compute_metrics=self.evaluate,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop)], **kwargs
        )

        trainer.train()
        metrics = trainer.evaluate()

        return metrics, trainer
