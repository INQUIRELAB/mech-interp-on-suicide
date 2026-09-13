"""
Run LLM annotation on UMD samples using Claude API.

Reads umd_annotation_sheet.xlsx, fills in llm_prediction, llm_confidence, llm_reasoning
using Anthropic Claude API, then saves the updated Excel.
"""

import pandas as pd
import sys
from pathlib import Path
from anthropic import Anthropic

# Configuration
API_KEY = None  # Set your key here or it will read from ANTHROPIC_API_KEY env var
MODEL = "claude-3-7-sonnet-20250219"  # or claude-3-opus-20240229 for better quality
MAX_TOKENS = 1000

PROMPT_TEMPLATE = """You are a clinical expert in suicide risk assessment.

Read the following Reddit post(s) and classify whether the person shows signs of suicidal ideation.

TEXT:
{text}

Respond in exactly this format (one line per field):
PREDICTION: [suicidal or not_suicidal]
CONFIDENCE: [high or medium or low]
REASONING: [one sentence explanation]

Consider:
- Explicit expressions of suicidal thoughts or plans
- Implicit signs like hopelessness, feeling trapped, being a burden
- Context matters (r/SuicideWatch posts are more likely to be suicidal)

Classify as "suicidal" only if there is clear evidence of suicidal ideation or intent.
"""

def run_claude_annotation(text, client):
    """Run Claude API on a single text sample."""
    prompt = PROMPT_TEMPLATE.format(text=text[:3000])  # Truncate for API

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        return parse_response(content)
    except Exception as e:
        print(f"  API Error: {e}")
        return {"prediction": "", "confidence": "", "reasoning": f"Error: {e}"}


def parse_response(response_text):
    """Parse Claude's structured response."""
    result = {"prediction": "", "confidence": "", "reasoning": ""}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("PREDICTION:"):
            result["prediction"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result



def _data_dir():
    """Data directory: $DATA_DIR if set, else <repo root>/data.

    Keeps this script runnable on any machine; the UMD files it reads are
    access-controlled and must be placed there by the user.
    """
    import os
    from pathlib import Path
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"

def main():
    input_path = _data_dir() / "umd_annotation_sheet.xlsx"
    output_path = _data_dir() / "umd_annotation_sheet_annotated.xlsx"

    # Load Excel
    print(f"Loading {input_path}...")
    df = pd.read_excel(input_path, engine='openpyxl')

    # Count samples to annotate
    to_annotate = df[df["llm_prediction"].isna() | (df["llm_prediction"] == "")]
    print(f"Samples to annotate: {len(to_annotate)}")

    if len(to_annotate) == 0:
        print("All samples already annotated!")
        return

    # Initialize Claude client
    import os
    api_key = API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable or API_KEY in script")
        return

    client = Anthropic(api_key=api_key)

    # Annotate each sample
    for idx, row in to_annotate.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Annotating sample_{row['sample_id']}...")

        result = run_claude_annotation(row["text"], client)

        df.at[idx, "llm_prediction"] = result["prediction"]
        df.at[idx, "llm_confidence"] = result["confidence"]
        df.at[idx, "llm_reasoning"] = result["reasoning"]

        # Check correctness
        if result["prediction"]:
            is_correct = "YES" if result["prediction"] == row["binary_label"] else "NO"
            df.at[idx, "llm_correct"] = is_correct
            print(f"  Prediction: {result['prediction']} | Truth: {row['binary_label']} | {is_correct}")

        # Save progress incrementally
        if (idx + 1) % 10 == 0:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"  [Progress saved]")

    # Final save
    df.to_excel(output_path, index=False, engine='openpyxl')

    # Print summary
    print("\n" + "="*50)
    print("ANNOTATION SUMMARY")
    print("="*50)

    annotated = df[df["llm_prediction"] != ""]
    accuracy = (annotated["llm_correct"] == "YES").sum() / len(annotated) * 100 if len(annotated) > 0 else 0

    print(f"Total annotated: {len(annotated)}")
    print(f"Accuracy: {accuracy:.1f}%")

    print("\nConfusion Matrix:")
    pd.crosstab(annotated["binary_label"], annotated["llm_prediction"],
                rownames=["Truth"], colnames=["Prediction"])

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
