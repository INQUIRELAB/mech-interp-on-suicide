"""Download models from HuggingFace in advance.

Usage:
    python download_models.py                    # Show model info
    python download_models.py llama-3.1-8b       # Download specific model
    python download_models.py qwen2.5-7b           # Download Qwen
    python download_models.py all                  # Download all three models

Authentication:
    Llama models require HuggingFace login:
        huggingface-cli login
    Or set HF_TOKEN environment variable.
"""

import sys

from src.models import ModelLoader, print_model_info


def download(model_name: str):
    """Download a single model."""
    loader = ModelLoader()
    try:
        loader.download_model(model_name)
        print(f"\n✓ {model_name} downloaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to download {model_name}: {e}")
        if "401" in str(e) or "gated" in str(e).lower():
            print("\nThis model requires authentication.")
            print("Run: huggingface-cli login")
            print("Or:  export HF_TOKEN=your_token")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print_model_info()
        print("\nUsage: python download_models.py <model_name>")
        print("       python download_models.py all")
        print("\nExamples:")
        print("  python download_models.py llama-3.1-8b")
        print("  python download_models.py qwen2.5-7b")
        print("  python download_models.py mistral-7b")
        return

    model_name = sys.argv[1].lower()

    if model_name == "all":
        models = ["llama-3.1-8b", "qwen2.5-7b", "mistral-7b"]
        for m in models:
            download(m)
        print("\n" + "=" * 60)
        print("All models downloaded!")
    else:
        download(model_name)


if __name__ == "__main__":
    main()
