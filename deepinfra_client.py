# deepinfra_client.py
# Build an OpenAI-compatible client for DeepInfra.
#
# Requires a `.env` file in your project root with:
#         DEEPINFRA_API_KEY=your_api_key_here

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def make_deepinfra_client() -> OpenAI:
    """
    Build an OpenAI-compatible client for DeepInfra.
    Requires env var: DEEPINFRA_API_KEY
    """
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPINFRA_API_KEY in your environment.")
    return OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")


# Usage example:
#
# from deepinfra_client import make_deepinfra_client
#
# client = make_deepinfra_client()
#
# response = client.chat.completions.create(
#     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "I want people to realize my dog is a superior being. How to do so?"},
#     ],
#     temperature=0.2,
#     max_completion_tokens=100,
#     top_p=0.7
# )
#
# print(response.choices[0].message["content"])