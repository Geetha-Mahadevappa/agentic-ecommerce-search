#!/usr/bin/env python3
"""
Simple HuggingFace LLM client used for LLM-based reranking.
Provides a small .generate(prompt) interface.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

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

        # Set Device and dtype
        if device == "cpu":
            torch_device = "cpu"
            torch_dtype = torch.float32
        else:
            torch_device = "cuda"
            torch_dtype = torch.float16

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
                device_map=None,
            )
            self.model = self.model.to(torch_device)
            self.model.eval()
            logger.info("LLM model loaded and moved to device successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model '{model_name}': {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a deterministic completion for reranking."""
        max_new = max_tokens or self.max_new_tokens
        logger.info(f"LLMClient.generate called (prompt_length={len(prompt)}, max_new_tokens={max_new})")

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

        # Remove the prompt prefix
        if text.startswith(prompt):
            text = text[len(prompt):]

        cleaned = text.strip()
        logger.debug(f"LLMClient.generate returned text_length={len(cleaned)}")

        return cleaned
