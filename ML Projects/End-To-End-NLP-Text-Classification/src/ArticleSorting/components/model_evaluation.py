from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader
import numpy as np
from ArticleSorting.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) :
        self.config = config
        
    
    def evaluate(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
        torch.cuda.empty_cache() # Empty cache

        # Loading data
        test_dataset = load_from_disk(self.config.test_data_path)
        #print(test_dataset)

        # DataLoader
        test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=4, **kwargs)

    
        #Loading the model 
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path).to(device)


        final_output = []
        total_acc_test = 0
        for b_idx, data in enumerate(test_dataloader):
            with torch.no_grad():
                for key, value in data.items():
                    data[key] = value.to(device)
                output = model(**data)
                output = output.logits.detach().cpu().numpy()
                final_output.extend(output)
                
        
        preds = np.vstack(final_output)
        preds = np.argmax(preds, axis=1)
        total_acc_test += sum(1 if preds[i] == test_dataset["labels"][i] else 0 for i in range(len(test_dataset)))
        Test_Accuracy = total_acc_test / len(test_dataset)
        
        print(f'Predictions : {preds}')
        print(f'Labels : {test_dataset["labels"]}')
        print(f'Test Accuracy: {Test_Accuracy: .3f}')
        
           
        df = pd.DataFrame([Test_Accuracy], index=['bert'])
        df.to_csv(self.config.metric_file_name, index=False)

        torch.cuda.empty_cache()
