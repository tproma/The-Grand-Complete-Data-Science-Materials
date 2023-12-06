from ArticleSorting.config.configuration import ConfigurationManager
from ArticleSorting.components.model_trainer import ModeTrainer
from ArticleSorting.logging import logger

class ModeTrainerTrainingPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModeTrainer(config= model_trainer_config)
        model_trainer.train()