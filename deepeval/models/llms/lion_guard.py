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
import tiktoken
from openai import BadRequestError

MAX_TOKENS_PER_CHUNK = 3000
OVERLAP_TOKENS = 200
EMBED_MODEL = "text-embedding-3-large"

# --- token utils ---
def get_tokenizer(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

enc = get_tokenizer(EMBED_MODEL)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def chunk_by_tokens(text: str, max_tokens=MAX_TOKENS_PER_CHUNK, overlap=OVERLAP_TOKENS):
    tokens = enc.encode(text)
    chunks, i = [], 0
    while i < len(tokens):
        window = tokens[i:i+max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap if max_tokens > overlap else max_tokens
    return chunks

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

        # decide if chunking is needed
        if count_tokens(prompt) <= MAX_TOKENS_PER_CHUNK:
            chunks = [prompt]
        else:
            chunks = chunk_by_tokens(prompt)

        merged_reasons = set()
        merged_score = 0

        for ch in chunks:
            try:
                response = self.openai_client.embeddings.create(
                    input=ch,
                    model=EMBED_MODEL,
                )
            except BadRequestError as e:
                raise RuntimeError(f"Embedding failed for chunk: {e}")

            embeddings = np.array([data.embedding for data in response.data])

            # Run LionGuard 2
            cat2scores = model.predict(embeddings)

            reasons = []
            for k, v in cat2scores.items():
                if v[0] >= 0.5:
                    reasons.append(k)

            if reasons:
                merged_score = 1  # once any chunk = 1, stick with it
                merged_reasons.update(reasons)

        if merged_score == 0:
            reason = "N/A"
        else:
            reason = ", ".join(sorted(merged_reasons))
            
        output = f'{{"reason": "{reason}", "score": {merged_score}}}'
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name
