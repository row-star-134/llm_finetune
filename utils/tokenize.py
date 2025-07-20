from transformers import GemmaTokenizer, AutoTokenizer


class CustomTokenizer:

    def __init__(self, model_name):
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenization(self, text):

        text = self.gemma_tokenizer.bos_token + text
        gemma_tokens = self.gemma_tokenizer(text, add_special_tokens=False, return_tensors="pt")

        return gemma_tokens

    def answer_tokenization(self, text):

        text =  text + self.gemma_tokenizer.eos_token
        gemma_tokens = self.gemma_tokenizer(text, add_special_tokens=False, return_tensors="pt")

        return gemma_tokens














