import json, os
import numpy as np
from ray import client
import torch
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
build_transformers_prefix_allowed_tokens_fn,
)

from deepeval.models import DeepEvalBaseLLM
from typing import Optional, Tuple, Union, Dict
from openai import OpenAI

class LionGuardModel(DeepEvalBaseLLM):
    def __init__(self, pretrained_model_name_or_path, OPENAI_API_KEY, device="auto", 
                 cache_dir=None):
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            #device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        self.name = pretrained_model_name_or_path
        self.model = model
        # Get OpenAI embeddings (users to input their own OpenAI API key)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def load_model(self):
        return self.model

    # schema should be a type class.
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> BaseModel:
        # Same as the previous example above
        model = self.load_model()

        response = self.openai_client.embeddings.create(
            input=prompt,
            model="text-embedding-3-large",
            )
        # embeddings: [1, 1536]
        embeddings = np.array([data.embedding for data in response.data])

        # Run LionGuard 2
        results = model.predict(embeddings)

        score = 0
        output = f'{{"reason": "n/a", "score": {score}}}'
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name
