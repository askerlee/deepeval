from together import Together
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from typing import Optional, Tuple, Union, Dict
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
import re, json

class TogetherModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, sys_prompt: str = "", enable_thinking: bool = True, temperature: float = 0.6):
        self.name = model_name
        self.sys_prompt = sys_prompt
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.TOGETHER_API_KEY)
        self.client = Together(api_key=self.api_key)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, sys_prompt: str=None, schema: Optional[BaseModel] = None) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": sys_prompt or self.sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        message = response.choices[0].message.content
        # Remove "<think>...</think>" 
        if self.enable_thinking:
            message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL).strip()
        json_result = json.loads(message)

        if schema:
            # Return valid JSON object according to the schema DeepEval supplied
            return schema(**json_result)
        else:
            return (message, 0)
        
    async def a_generate(self, prompt: str, sys_prompt: str=None, schema: Optional[BaseModel] = None) -> BaseModel:
        return self.generate(prompt, sys_prompt, schema)

    def get_model_name(self):
        return self.name
