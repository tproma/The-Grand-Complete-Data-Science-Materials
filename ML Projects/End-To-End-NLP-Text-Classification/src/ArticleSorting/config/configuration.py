from ArticleSorting.constants import *
from ArticleSorting.utils.common import read_yaml, create_directories
from ArticleSorting.entity import (DataIngestionConfig,
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModeTrainerConfig,
                                   ModelEvaluationConfig

)

class ConfigurationManager:
    def __init__(
                self, 
                config_filepath = CONFIG_FILE_PATH,
                params_filepath = PARAMS_FILE_PATH):
        
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
          config = self.config.data_ingestion

          create_directories([config.root_dir])

          data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                source_URL= config.source_URL,
                local_data_file = config.local_data_file,
                unzip_dir = config.unzip_dir
                )
          
          return data_ingestion_config
    

    def get_data_validation_config(self) ->DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            ALL_REQUIRED_FILES= config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    

    def get_data_transformation_config(self) ->DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path= config.data_path,
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config


    def get_model_trainer_config(self) -> ModeTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModeTrainerConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_ckpt = config.model_ckpt,
            output_dir = params.output_dir,
            learning_rate  = params.learning_rate,
            per_device_train_batch_size  = params.per_device_train_batch_size,
            per_device_eval_batch_size  = params.per_device_eval_batch_size,
            num_train_epochs  = params.num_train_epochs,
            weight_decay= params.weight_decay,
            eval_steps= params.eval_steps,
            evaluation_strategy= params.evaluation_strategy,
            save_strategy = params.save_strategy,
            load_best_model_at_end= params.load_best_model_at_end
        )

        return model_trainer_config
    


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
      
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            test_data_path = config.test_data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name

        )
        return model_evaluation_config

