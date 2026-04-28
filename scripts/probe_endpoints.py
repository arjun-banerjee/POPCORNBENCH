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
        "FW-GLM-5-1",
        "openai_chat",
        "https://popcorn-foundry-resource.openai.azure.com/openai/v1/",
        "POPCORN_AZURE_KEY",
        "FW-GLM-5-1",
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


# A throwaway tool schema in the same shape the agent sends. If Azure rejects
# `tools=` for a given model/deployment, this probe will surface it.
_DUMMY_TOOL_RESPONSES = {
    "type": "function",
    "name": "noop",
    "description": "Does nothing.",
    "parameters": {"type": "object", "properties": {}},
}
_DUMMY_TOOL_CHAT = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _try(label, fn):
    try:
        fn()
        return f"  {label}: OK"
    except Exception as e:
        return f"  {label}: FAIL ({type(e).__name__}: {str(e)[:120]})"


def probe(name, api_kind, base_url, key_env, model):
    key = os.environ.get(key_env)
    if not key:
        return False, [f"env var {key_env} not set"]
    client = OpenAI(api_key=key, base_url=base_url)
    results = []
    if api_kind == "openai":
        results.append(_try(
            "bare",
            lambda: client.responses.create(model=model, input="ping"),
        ))
        results.append(_try(
            "with instructions",
            lambda: client.responses.create(
                model=model,
                instructions="Be terse.",
                input=[{"role": "user", "content": "ping"}],
            ),
        ))
        results.append(_try(
            "with tools",
            lambda: client.responses.create(
                model=model,
                instructions="Be terse.",
                input=[{"role": "user", "content": "ping"}],
                tools=[_DUMMY_TOOL_RESPONSES],
            ),
        ))
        results.append(_try(
            "with tools + reasoning.effort=low",
            lambda: client.responses.create(
                model=model,
                instructions="Be terse.",
                input=[{"role": "user", "content": "ping"}],
                tools=[_DUMMY_TOOL_RESPONSES],
                reasoning={"effort": "low"},
            ),
        ))
    else:
        results.append(_try(
            "bare",
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=16,
            ),
        ))
        results.append(_try(
            "with tools",
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                tools=[_DUMMY_TOOL_CHAT],
                tool_choice="auto",
                max_tokens=16,
            ),
        ))
    ok = all("OK" in r for r in results)
    return ok, results


def main():
    load_dotenv(override=False)
    rows = []
    for name, api_kind, base_url, key_env, model in PROBES:
        print(f"\n{name}  ({api_kind})")
        ok, detail = probe(name, api_kind, base_url, key_env, model)
        if isinstance(detail, list):
            for line in detail:
                print(line)
        else:
            print(f"  {detail}")
        rows.append((name, ok, detail))
    n_ok = sum(1 for _, ok, _ in rows if ok)
    print(f"\n{n_ok}/{len(rows)} endpoints fully passing all probes")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
