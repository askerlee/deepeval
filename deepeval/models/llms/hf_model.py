import json
import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
build_transformers_prefix_allowed_tokens_fn,
)

from deepeval.models import DeepEvalBaseLLM
from typing import Optional, Tuple, Union, Dict

class HFModel(DeepEvalBaseLLM):
    def __init__(self, pretrained_model_name_or_path, sys_prompt="", device="auto", 
                 enable_thinking=True, cache_dir=None):
        quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                              )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )

        self.name = pretrained_model_name_or_path
        self.model = model_4bit
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.enable_thinking = enable_thinking

    def load_model(self):
        return self.model

    # schema should be a type class.
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel:
        # Same as the previous example above
        model = self.load_model()
        messages=[
            {"role": "system", "content": self.sys_prompt},
            {"role": "user",   "content": prompt}
        ],
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking # Switches between thinking and non-thinking modes.
        )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=5000,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        with torch.no_grad():
            output_dict = pipeline(text, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0][0]["generated_text"][len(text[0]):]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name
