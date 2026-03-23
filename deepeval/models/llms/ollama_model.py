import json
import re
from ollama import Client, AsyncClient, ChatResponse
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


def _extract_json_dict(content: str) -> Dict:
    start = content.find("{")
    end = content.rfind("}") + 1

    if end == 0 and start != -1:
        content = content + "}"
        end = len(content)

    if start == -1 or end == 0 or end <= start:
        raise ValueError("No JSON object found in Ollama response.")

    json_str = content[start:end]
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    return json.loads(json_str)


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        max_num_tokens: int = 4096,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_MODEL_NAME
        )
        self.base_url = (
            base_url
            or KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_BASE_URL)
            or "http://localhost:11434"
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.max_num_tokens = max_num_tokens
        super().__init__(model_name)

    @staticmethod
    def _parse_schema_response(content: str, schema: BaseModel):
        try:
            return schema.model_validate_json(content)
        except Exception as first_error:
            try:
                return schema.model_validate(_extract_json_dict(content))
            except Exception:
                raise first_error

    ###############################################
    # Other generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        response: ChatResponse = chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={ "temperature": self.temperature, "num_predict": self.max_num_tokens, 
                      # Not sure if prompt KV cache is enabled by default.
                      # Enable KV cache might speed up prefilling the common prefix of the harm template. 
                      "cache_prompt": True},
            think=False,
        )

        if schema:
            try:
                result = self._parse_schema_response(
                    response.message.content, schema
                )
                return (result, 0)
            except Exception as e:
                # Bug in llama guard encountered.
                # Always set such occasions to safe, as such occasions are rare, and 
                # the output from llama guard is already probably meaningless.
                response.message.content = '{"reason": "buggy output, regarded as safe", "score": 0}'
                result = schema.model_validate_json(response.message.content)
                return (result, 0)
        else:
            return (response.message.content, 0)

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        response: ChatResponse = await chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={"temperature": self.temperature, 
                     "num_predict": self.max_num_tokens},
            think=False,
        )

        if schema:
            try:
                result = self._parse_schema_response(
                    response.message.content, schema
                )
                return (result, 0)
            except Exception as e:
                # Bug in llama guard encountered.
                # Always set such occasions to safe, as such occasions are rare, and
                # the output from llama guard is already probably meaningless.
                response.message.content = '{"reason": "buggy output, regarded as safe", "score": 0}'
                result = schema.model_validate_json(response.message.content)
                return (result, 0)
        else:
            return (response.message.content, 0)

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return Client(host=self.base_url)
        else:
            return AsyncClient(host=self.base_url)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
