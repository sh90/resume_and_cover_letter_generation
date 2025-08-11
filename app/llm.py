from dotenv import load_dotenv
import os
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional

load_dotenv()  # load .env variables automatically

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class GenConfig:
    model: str = os.getenv("GEN_MODEL", "gpt-4o-mini")
    max_tokens: int = 600
    temperature: float = 0.1

def generate_text(prompt: str, cfg: Optional[GenConfig] = None) -> str:
    cfg = cfg or GenConfig()
    resp = client.responses.create(
        model=cfg.model,
        input=prompt,
        max_output_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    return resp.output_text
