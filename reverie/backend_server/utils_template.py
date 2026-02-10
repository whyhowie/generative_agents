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

# Other settings used by the simulation server
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
