#!/usr/bin/env python3
"""
Simple HuggingFace LLM client used for LLM-based reranking.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from logging_config import setup_logging
setup_logging("logs/agent_runtime.log")
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
    ):
        logger.info(f"Initializing LLMClient with model '{model_name}' on device '{device}'")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # For reranking
        self.do_sample = False
        self.temperature = None
        self.top_k = None
        self.top_p = None

        # Device selection
        if device == "cpu":
            hf_device = -1
            torch_dtype = torch.float32
            torch_device = "cpu"
        else:
            hf_device = 0
            torch_dtype = torch.float16
            torch_device = "cuda"

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for model '{model_name}': {e}")
            raise

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            self.model.eval()
            self.model = self.model.to(torch_device)
            logger.info("LLM model loaded and moved to device successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model '{model_name}': {e}")
            raise

        # Build pipeline using the already-loaded model
        try:
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=hf_device,
            )
            logger.info("LLM text-generation pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM pipeline: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a deterministic completion for reranking."""
        max_new = max_tokens or self.max_new_tokens
        logger.info(
            f"LLMClient.generate called (prompt_length={len(prompt)}, max_new_tokens={max_new})"
        )

        try:
            with torch.no_grad():
                out = self.pipe(
                    prompt,
                    max_new_tokens=max_new,
                    do_sample=self.do_sample,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

        text = out[0]["generated_text"]

        # Remove the prompt prefix
        if text.startswith(prompt):
            text = text[len(prompt):]

        cleaned = text.strip()
        logger.debug(f"LLMClient.generate returned text_length={len(cleaned)}")

        return cleaned
