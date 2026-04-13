#!/usr/bin/env python3
"""Query a running vLLM OpenAI-compatible completions server with a prompt file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt_file",
        type=Path,
        help="Path to a text file containing the exact prompt to send.",
    )
    parser.add_argument(
        "--inference-model-name",
        required=True,
        help="Model name exposed by vLLM, for example 'myadapter'.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to LOCAL_SERVER_* env vars.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY"),
        help="API key for the OpenAI-compatible server.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional stop sequence. Can be passed multiple times.",
    )
    return parser.parse_args()


def default_base_url() -> str:
    endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
    port = os.getenv("LOCAL_SERVER_PORT", "8000")
    return f"http://{endpoint}:{port}/v1"


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")

    client = OpenAI(
        base_url=args.base_url or default_base_url(),
        api_key=args.api_key,
    )

    response = client.completions.create(
        model=args.inference_model_name,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=args.stop,
    )

    print(response.choices[0].text, end="")


if __name__ == "__main__":
    main()
