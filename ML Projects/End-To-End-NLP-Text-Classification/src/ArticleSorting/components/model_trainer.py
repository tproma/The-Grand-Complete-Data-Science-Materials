from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

import evaluate
import numpy as np
import os
from ArticleSorting.entity import ModeTrainerConfig

class ModeTrainer:
    def __init__(self, config: ModeTrainerConfig) :
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        # Empty cache
        torch.cuda.empty_cache()

        # Loading data
        train_dataset = load_from_disk(self.config.train_data_path)
        test_dataset = load_from_disk(self.config.test_data_path)
    
        # DataLoader
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=4)
        eval_dataloader = DataLoader(dataset=test_dataset, batch_size=4)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        id2label = {0: "business", 1: "entertainment", 2: "politics", 3: "sport", 4: "tech"}
        label2id = {"business": 0, "entertainment": 1, "politics": 2, "sport": 3, "tech": 4 }
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_ckpt,
            num_labels=5,
            id2label=id2label, 
            label2id=label2id
            ).to(device)
        
        
        accuracy = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        
                
        training_args = TrainingArguments(
            output_dir="bert-base-cased",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_steps = 50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,

        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

         ## Save model
        model.save_pretrained(os.path.join(self.config.root_dir,"bert-base-uncased-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))

