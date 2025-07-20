from peft import get_peft_model
from transformers import Gemma3ForCausalLM


class BaseModel:
    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name

    def load_model(self):
        print(self.model_name)
        base_model = Gemma3ForCausalLM.from_pretrained(self.model_name, ignore_mismatched_sizes=True).to(self.device)
        return base_model


class LoraModel:
    def __init__(self, lora_config, base_model):
        self.lora_config = lora_config
        self.base_model = base_model

    def load_lora_model(self):
        lora_model = get_peft_model(self.base_model, self.lora_config)
        lora_model.print_trainable_parameters()
        return lora_model

