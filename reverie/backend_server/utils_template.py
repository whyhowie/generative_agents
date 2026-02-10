"""
Local configuration for LLM providers.

You can also set environment variables instead of editing this file:
- LLM_PROVIDER: "gemini" or "openai"
- LLM_MODEL: default text model
- LLM_EMBEDDING_MODEL: default embedding model
- GEMINI_API_KEY / GOOGLE_API_KEY
- OPENAI_API_KEY
"""

# Default provider: "gemini" or "openai"
llm_provider = "gemini"

# Optional overrides for model selection
llm_model = None
llm_embedding_model = None

# API keys (optional if you use environment variables)
gemini_api_key = ""
openai_api_key = ""
