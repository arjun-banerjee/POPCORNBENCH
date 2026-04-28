"""Quick liveness test for every model endpoint in the sweep.

Sends one trivial request through the OpenAI Python SDK at each
(base_url, api_kind, model) tuple and reports OK / fail.

Run from repo root:
    uv run python scripts/probe_endpoints.py
"""

from __future__ import annotations

import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAI


PROBES = [
    # (display_name, api_kind, base_url, key_env, model_id_to_send)
    (
        "gpt-5.4-pro",
        "openai",
        "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1/",
        "TEJAS_AZURE_KEY",
        "gpt-5.4-pro",
    ),
    (
        "gpt-5.5",
        "openai",
        "https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1/",
        "TEJAS_AZURE_KEY",
        "gpt-5.5",
    ),
    (
        "DeepSeek-R1",
        "openai_chat",
        "https://thava-openai.services.ai.azure.com/models",
        "THAVA_AZURE_KEY",
        "DeepSeek-R1",
    ),
    (
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "openai_chat",
        "https://thava-openai.services.ai.azure.com/models",
        "THAVA_AZURE_KEY",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
    ),
    (
        "Kimi-K2.6",
        "openai_chat",
        "https://thava-openai.services.ai.azure.com/models",
        "THAVA_AZURE_KEY",
        "Kimi-K2.6",
    ),
]


def probe(name, api_kind, base_url, key_env, model):
    key = os.environ.get(key_env)
    if not key:
        return False, f"env var {key_env} not set"
    client = OpenAI(api_key=key, base_url=base_url)
    try:
        if api_kind == "openai":
            r = client.responses.create(model=model, input="ping")
            text = ""
            for item in r.output:
                if getattr(item, "type", None) == "message":
                    for c in item.content:
                        text += getattr(c, "text", "") or ""
            return True, (text or "<no text>")[:80]
        else:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=16,
            )
            return True, (r.choices[0].message.content or "<no text>")[:80]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    load_dotenv(override=False)
    rows = []
    width = max(len(p[0]) for p in PROBES)
    for name, api_kind, base_url, key_env, model in PROBES:
        ok, detail = probe(name, api_kind, base_url, key_env, model)
        status = "OK  " if ok else "FAIL"
        print(f"{status}  {name:<{width}}  {api_kind:<11}  {detail}")
        rows.append((name, ok, detail))
    n_ok = sum(1 for _, ok, _ in rows if ok)
    print(f"\n{n_ok}/{len(rows)} endpoints reachable")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
