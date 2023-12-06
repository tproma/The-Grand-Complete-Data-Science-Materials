import os
from ArticleSorting.logging import logger
from ArticleSorting.entity import DataTransformationConfig

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        

    def encode_categories(self):

        df = pd.read_csv(self.config.data_path)
        df['encoded_label'] = df['Category'].astype('category').cat.codes

        '''Spliting the Data
        # Training dataset
        #train_data = df.sample(frac=0.8, random_state=42)
        # Testing dataset
        #test_data = df.drop(train_data.index)
        '''

        # Split train , test and validatio dataset
        np.random.seed(112)
        df_train, df_test, df_val = np.split(df.sample(frac=1, random_state=35),
                                     [int(0.7*len(df)), int(0.8*len(df))])

        print(len(df_train), len(df_test), len(df_val)) 
        

        # Convert pyhton dataframe to Hugging Face arrow dataset
        hg_train_data = Dataset.from_pandas(df_train)
        hg_test_data = Dataset.from_pandas(df_test)
        hg_val_data = Dataset.from_pandas(df_val)
        print(hg_train_data, hg_test_data, hg_val_data)
        return hg_train_data, hg_test_data, hg_val_data
   


    def tokenize_dataset(self, data):
        return self.tokenizer(data["Text"],
                     max_length=512,
                     truncation=True,
                     padding="max_length")
    
    
    def convert(self):
       
        hg_train_data, hg_test_data, hg_val_data = self.encode_categories()

        # Tokenize the dataset
        dataset_train = hg_train_data.map(self.tokenize_dataset)
        dataset_test = hg_test_data.map(self.tokenize_dataset)
        dataset_val = hg_val_data.map(self.tokenize_dataset)

        # Remove the review and index columns because it will not be used in the model
        dataset_train = dataset_train.remove_columns(["ArticleId", "Text", "Category", "__index_level_0__"])
        dataset_test = dataset_test.remove_columns(["ArticleId", "Text", "Category", "__index_level_0__"])
        dataset_val = dataset_val.remove_columns(["ArticleId", "Text", "Category", "__index_level_0__"])

        # Rename label to labels because the model expects the name labels
        dataset_train = dataset_train.rename_column("encoded_label", "labels")
        dataset_test = dataset_test.rename_column("encoded_label", "labels")
        dataset_val = dataset_val.rename_column("encoded_label", "labels")

        # Change the format to PyTorch tensors
        dataset_train.set_format("torch")
        dataset_test.set_format("torch")
        dataset_val.set_format("torch")

        # Take a look at the data
        print(dataset_train)
        print(dataset_test)
        print(dataset_val)

        # Saving the datasets
        dataset_train.save_to_disk(os.path.join(self.config.root_dir,"Train BBC dataset"))
        dataset_test.save_to_disk(os.path.join(self.config.root_dir,"Test BBC dataset"))
        dataset_val.save_to_disk(os.path.join(self.config.root_dir,"Validation BBC dataset"))