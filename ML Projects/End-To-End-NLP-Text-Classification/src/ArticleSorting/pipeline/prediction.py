from ArticleSorting.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline



class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        

        pipe = pipeline("text-classification", model=self.config.model_path,tokenizer=tokenizer)

        print("Text: ")
        print(text)

        output = pipe(text)
        print("\nText Category:")
        print(output)

        return output