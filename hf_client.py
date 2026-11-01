"""
CodeAlpha Artificial Intelligence Internship
TASK 2: Chatbot for FAQs - Hugging Face Inference API Client
Enables the chatbot to answer open-domain and general AI questions using Hugging Face LLMs.
"""

import os
import json
from typing import Dict, Any, Optional
import requests

# Popular high-performance models supported on Hugging Face Inference API
SUPPORTED_HF_MODELS = {
    "Llama 3.2 (3B)": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen 2.5 (72B)": "Qwen/Qwen2.5-72B-Instruct",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
    "Phi-3 Mini": "microsoft/Phi-3-mini-4k-instruct",
}

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class HuggingFaceClient:
    """
    Communicates with Hugging Face Serverless Inference API / Router
    to generate intelligent conversational responses to any query.
    """

    def __init__(self, api_token: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_token = (
            api_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or ""
        ).strip()
        self.model_name = model_name

    def set_token(self, token: str):
        """Updates the Hugging Face API token."""
        self.api_token = (token or "").strip()

    def set_model(self, model_name: str):
        """Updates target model."""
        self.model_name = model_name

    def is_configured(self) -> bool:
        """Returns True if an API token has been provided."""
        return bool(self.api_token and len(self.api_token) > 5)

    def generate_response(
        self,
        user_query: str,
        system_context: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Sends query to Hugging Face router / chat completion API.
        Returns a dictionary with result, success status, model, and metadata.
        """
        if not self.is_configured():
            return {
                "success": False,
                "response": (
                    "To enable Hugging Face to answer any question, please enter your free Hugging Face API token "
                    "in the sidebar! (You can create one for free at https://huggingface.co/settings/tokens)."
                ),
                "model": "None",
                "error": "No Hugging Face token provided.",
            }

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # Build messages for conversational format
        sys_msg = (
            system_context
            or "You are CodeAlpha AI Assistant, a helpful, precise, and friendly AI chatbot. "
            "Answer the user's question clearly and informatively."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query},
        ]

        # 1. Try OpenAI-Compatible Router Endpoint
        router_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(router_url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return {
                    "success": True,
                    "response": content,
                    "model": self.model_name,
                    "provider": "Hugging Face Router",
                }
            elif resp.status_code == 401:
                return {
                    "success": False,
                    "response": "⚠️ Invalid Hugging Face token. Please check that your token was copied correctly from huggingface.co/settings/tokens.",
                    "model": self.model_name,
                    "error": "401 Unauthorized",
                }
        except Exception as e:
            pass

        # 2. Fallback to direct model inference endpoint
        direct_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        direct_payload = {
            "inputs": f"<|system|>\n{sys_msg}</s>\n<|user|>\n{user_query}</s>\n<|assistant|>\n",
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        try:
            resp = requests.post(direct_url, headers=headers, json=direct_payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    content = data[0].get("generated_text", "").strip()
                    return {
                        "success": True,
                        "response": content or "No response generated.",
                        "model": self.model_name,
                        "provider": "Hugging Face Model Endpoint",
                    }
            elif resp.status_code == 503:
                return {
                    "success": False,
                    "response": "⏳ Model is currently loading on Hugging Face servers. Please wait 10 seconds and try again!",
                    "model": self.model_name,
                    "error": "Model loading",
                }
            else:
                return {
                    "success": False,
                    "response": f"Hugging Face API returned status {resp.status_code}: {resp.text[:150]}",
                    "model": self.model_name,
                    "error": resp.text,
                }
        except Exception as e:
            return {
                "success": False,
                "response": f"Connection to Hugging Face API failed: {e}",
                "model": self.model_name,
                "error": str(e),
            }


# Singleton instance
hf_client = HuggingFaceClient()
