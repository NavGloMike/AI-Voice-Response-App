import os
from dotenv import load_dotenv, find_dotenv

# Check if the .env file exists and load it
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

record_audio = os.getenv("RECORD_AUDIO", "false").lower() in ("true", "1", "t", "y", "yes")
debug_on = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "y", "yes")
max_tokens = os.getenv("MAX_TOKENS", "1000")  # Provide a default value
temperature = int(os.getenv("CONFIG_TEMP", "0"))  # Provide a default value
timeout = int(os.getenv("CONFIG_TIMEOUT", "30"))  # Provide a default value

epUrl = os.getenv("EPIM_FETCH_URL", "")
intakeUrl = os.getenv("INTAKE_FETCH_URL", "")

# EPIM configs
epim_config_url = os.getenv("EPIM_GET_CONFIG_URL", "")
epim_tier_config_url = os.getenv("EPIM_GET_TIER_CONFIG_URL", "")
epim_violation_qa_url = os.getenv("EPIM_GET_VIOLATION_QA_URL", "")
epim_ask_question_url = os.getenv("EPIM_GET_ASK_QUESTION_URL", "")