"""
Provider-agnostic gateway for LLM text generation and embeddings.
Defaults to Gemini, but can fall back to OpenAI.
"""
import os
from typing import Any, Dict, Optional


def _get_utils_attr(name: str, default: Optional[str] = None) -> Optional[str]:
  try:
    import utils
    return getattr(utils, name, default)
  except Exception:
    return default


def _get_provider() -> str:
  provider = os.getenv("LLM_PROVIDER") or _get_utils_attr("llm_provider", "gemini")
  return str(provider).strip().lower()


def _get_default_model(provider: str) -> str:
  env_model = os.getenv("LLM_MODEL")
  if env_model:
    return env_model
  utils_model = _get_utils_attr("llm_model")
  if utils_model:
    return utils_model
  if provider == "openai":
    return "gpt-3.5-turbo"
  return "gemini-1.5-pro"


def _get_default_embedding_model(provider: str) -> str:
  env_model = os.getenv("LLM_EMBEDDING_MODEL")
  if env_model:
    return env_model
  utils_model = _get_utils_attr("llm_embedding_model")
  if utils_model:
    return utils_model
  if provider == "openai":
    return "text-embedding-ada-002"
  return "text-embedding-004"


def _normalize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if not params:
    return {}
  return {k: v for k, v in params.items() if v is not None}


def _coerce_stop_sequences(stop: Any) -> Optional[list]:
  if stop is None:
    return None
  if isinstance(stop, (list, tuple)):
    return list(stop)
  return [str(stop)]


def _openai_api_key() -> Optional[str]:
  return os.getenv("OPENAI_API_KEY") or _get_utils_attr("openai_api_key")


def _gemini_api_key() -> Optional[str]:
  return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or _get_utils_attr("gemini_api_key")


def _openai_request(prompt: str, model: str, params: Dict[str, Any]) -> str:
  import openai

  api_key = _openai_api_key()
  if not api_key:
    raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or openai_api_key.")
  openai.api_key = api_key

  is_completion_model = model.startswith("text-") or model.startswith("davinci") or model.startswith("curie")
  if is_completion_model:
    response = openai.Completion.create(
      model=model,
      prompt=prompt,
      temperature=params.get("temperature"),
      max_tokens=params.get("max_tokens"),
      top_p=params.get("top_p"),
      frequency_penalty=params.get("frequency_penalty"),
      presence_penalty=params.get("presence_penalty"),
      stream=params.get("stream", False),
      stop=params.get("stop"),
    )
    return response.choices[0].text

  response = openai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=params.get("temperature"),
    max_tokens=params.get("max_tokens"),
    top_p=params.get("top_p"),
    frequency_penalty=params.get("frequency_penalty"),
    presence_penalty=params.get("presence_penalty"),
    stop=params.get("stop"),
  )
  return response["choices"][0]["message"]["content"]


def _extract_gemini_text(response: Any) -> str:
  text = getattr(response, "text", None)
  if text:
    return text
  try:
    parts = response.candidates[0].content.parts
    return "".join([p.text for p in parts if hasattr(p, "text")])
  except Exception:
    return ""


def _gemini_request(prompt: str, model: str, params: Dict[str, Any]) -> str:
  from google import genai
  from google.genai import types

  api_key = _gemini_api_key()
  if not api_key:
    raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY or gemini_api_key.")

  client = genai.Client(api_key=api_key)
  config = types.GenerateContentConfig(
    temperature=params.get("temperature"),
    top_p=params.get("top_p"),
    max_output_tokens=params.get("max_tokens"),
    stop_sequences=_coerce_stop_sequences(params.get("stop")),
  )
  response = client.models.generate_content(
    model=model,
    contents=prompt,
    config=config,
  )
  return _extract_gemini_text(response)


def llm_request(prompt: str, model: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
  provider = _get_provider()
  model = model or _get_default_model(provider)
  params = _normalize_params(params)

  if provider == "openai":
    return _openai_request(prompt, model, params)
  if provider == "gemini":
    return _gemini_request(prompt, model, params)

  raise ValueError(f"Unsupported LLM provider: {provider}")


def llm_embedding(text: str, model: Optional[str] = None) -> list:
  provider = _get_provider()
  model = model or _get_default_embedding_model(provider)

  if provider == "openai":
    import openai
    api_key = _openai_api_key()
    if not api_key:
      raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or openai_api_key.")
    openai.api_key = api_key
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

  if provider == "gemini":
    from google import genai
    api_key = _gemini_api_key()
    if not api_key:
      raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY or gemini_api_key.")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(model=model, contents=[text])
    return response.embeddings[0].values

  raise ValueError(f"Unsupported LLM provider: {provider}")
