"""
``era-memory`` command-line entry point.

Subcommands:
  setup    Interactively download a local embedding model from Hugging Face and cache it,
           so era-memory does semantic search offline with no endpoint.
  status   Show the current embedder situation (endpoint / cached models / fastembed).

This is the *only* place that initiates a download by default (plus the explicit
``build_memory(embedder="auto")``). A bare import or ``build_memory()`` never downloads.
"""

from __future__ import annotations

import argparse
import os
import sys


def _print_status() -> None:
    from .adapters.fastembed import (
        SUPPORTED_MODELS,
        default_cache_dir,
        fastembed_available,
        is_cached,
    )

    url = os.environ.get("MEMORY_EMBEDDING_URL")
    print("era-memory embedder status")
    print("-" * 40)
    print(f"Endpoint (MEMORY_EMBEDDING_URL): {url or '(not set)'}")
    print(f"fastembed installed:            {fastembed_available()}")
    print(f"Model cache dir:                {default_cache_dir()}")
    print("Local models:")
    any_cached = False
    for spec in SUPPORTED_MODELS.values():
        cached = is_cached(spec)
        any_cached = any_cached or cached
        mark = "✓ cached" if cached else "— not downloaded"
        print(f"  {spec.key:<12} {spec.fastembed_name:<40} dim={spec.dimensions:<5} {mark}")
    if not url and not any_cached:
        print("\nNo real embedder configured → search would use the non-semantic dev embedder.")
        print("Run `era-memory setup` to download one.")


def _cmd_setup(args: argparse.Namespace) -> int:
    from .adapters.fastembed import (
        DEFAULT_MODEL_KEY,
        SUPPORTED_MODELS,
        default_cache_dir,
        download_model,
        fastembed_available,
        first_cached_model,
        is_cached,
    )

    if os.environ.get("MEMORY_EMBEDDING_URL"):
        print("MEMORY_EMBEDDING_URL is set — era-memory will use that endpoint. Nothing to do.")
        return 0

    if not fastembed_available():
        print(
            "The local embedder needs the [localembed] extra.\n"
            "Install it with:\n\n    pip install 'era-memory[localembed]'\n",
            file=sys.stderr,
        )
        return 1

    already = first_cached_model()
    if already is not None and not args.force:
        print(f"A local model is already cached: {already.fastembed_name} (dim {already.dimensions}).")
        print("Use it with build_memory(...) directly, or pass --force to add another.")
        return 0

    # Resolve which model to fetch.
    model_key = args.model
    if model_key is None:
        if not sys.stdin.isatty() or args.yes:
            model_key = DEFAULT_MODEL_KEY  # non-interactive default
        else:
            model_key = _prompt_model_choice()
            if model_key is None:
                print("Aborted.")
                return 1
    if model_key not in SUPPORTED_MODELS:
        print(f"Unknown model '{model_key}'. Choices: {', '.join(SUPPORTED_MODELS)}", file=sys.stderr)
        return 2

    spec = SUPPORTED_MODELS[model_key]
    if is_cached(spec) and not args.force:
        print(f"{spec.fastembed_name} is already cached (dim {spec.dimensions}).")
        return 0

    if sys.stdin.isatty() and not args.yes:
        ans = input(
            f"Download {spec.fastembed_name} (~{spec.approx_size_mb} MB, {spec.license}, "
            f"dim {spec.dimensions}) from Hugging Face? [Y/n] "
        ).strip().lower()
        if ans not in ("", "y", "yes"):
            print("Aborted.")
            return 1

    print(f"Downloading {spec.fastembed_name} → {default_cache_dir()} …")
    download_model(spec)
    print(
        f"Done. Cached '{spec.key}' (dim {spec.dimensions}).\n"
        "era-memory will now use it automatically — e.g. build_memory(tier=0, db_path='memory.db')."
    )
    return 0


def _prompt_model_choice() -> "str | None":
    from .adapters.fastembed import DEFAULT_MODEL_KEY, SUPPORTED_MODELS

    keys = list(SUPPORTED_MODELS)
    print("Choose a local embedding model to download:")
    for i, key in enumerate(keys, 1):
        spec = SUPPORTED_MODELS[key]
        default = " (default)" if key == DEFAULT_MODEL_KEY else ""
        print(
            f"  {i}) {key}{default}: {spec.fastembed_name} — dim {spec.dimensions}, "
            f"~{spec.approx_size_mb} MB, {spec.license}"
        )
    raw = input(f"Enter 1-{len(keys)} [default {keys.index(DEFAULT_MODEL_KEY) + 1}]: ").strip()
    if raw == "":
        return DEFAULT_MODEL_KEY
    if raw.isdigit() and 1 <= int(raw) <= len(keys):
        return keys[int(raw) - 1]
    if raw in SUPPORTED_MODELS:
        return raw
    return None


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="era-memory", description="era-memory utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="download & cache a local embedding model")
    p_setup.add_argument("--model", choices=["bge-small", "mxbai-large"], help="skip the prompt")
    p_setup.add_argument("--yes", "-y", action="store_true", help="assume yes (non-interactive)")
    p_setup.add_argument("--force", action="store_true", help="download even if one is cached")
    p_setup.set_defaults(func=_cmd_setup)

    p_status = sub.add_parser("status", help="show embedder configuration")
    p_status.set_defaults(func=lambda _a: (_print_status() or 0))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
