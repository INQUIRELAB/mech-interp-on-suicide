"""
Run LLM annotation on UMD samples using Zhipu GLM API.

Reads umd_annotation_sheet.xlsx, fills in llm_prediction, llm_confidence, llm_reasoning
using Zhipu GLM-4 API, then saves the updated Excel.

Install: pip install zhipuai
Get API key: https://open.bigmodel.cn/
"""

import sys
import io
# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import sys
import os
import time
from pathlib import Path
from zhipuai import ZhipuAI

# Configuration
API_KEY = os.environ.get("GLM_API_KEY", "")  # set GLM_API_KEY or ZHIPUAI_API_KEY in env
# Model preference, best first. We probe these at startup and use the first that actually
# responds. Paid tiers (glm-4.6/4.5) return error 1113 "insufficient balance" until the
# account is topped up at open.bigmodel.cn; glm-4.5-flash is FREE and always works, so the
# script runs today on flash and auto-upgrades to glm-4.6 the moment balance is added.
# Set GLM_MODEL to pin a specific model and skip probing.
MODEL_PREFERENCE = ["glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash"]
MAX_RETRIES = 3  # transient errors (rate limit / network) are retried with backoff


def pick_model(client):
    """Return the first model in MODEL_PREFERENCE that the account can actually call."""
    pinned = os.environ.get("GLM_MODEL")
    if pinned:
        return pinned
    for model in MODEL_PREFERENCE:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=600,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if r.choices[0].message.content is not None:
                return model
        except Exception as e:
            s = str(e)
            reason = "no balance" if "1113" in s else ("not found" if "1211" in s else s[:50])
            print(f"  [skip] {model}: {reason}")
    raise RuntimeError("No usable GLM model found for this API key.")

PROMPT_TEMPLATE = """You are a clinical expert in suicide risk assessment.

Read the following Reddit post(s) and classify whether the person shows signs of suicidal ideation.

TEXT:
{text}

Respond in EXACTLY this format (one line per field):
PREDICTION: [suicidal or not_suicidal]
CONFIDENCE: [high or medium or low]
REASONING: [one sentence explanation]

Classification criteria:
- Explicit expressions of suicidal thoughts or plans → suicidal
- Implicit signs like hopelessness, feeling trapped, being a burden → possibly suicidal
- Context matters (r/SuicideWatch posts are more likely to be suicidal)

Only classify as "suicidal" if there is clear evidence of suicidal ideation or intent.
"""


def call_glm(client, prompt, model):
    """Call Zhipu GLM API using SDK, with retries on transient errors.

    GLM-4.5 is a hybrid reasoning model: with thinking enabled it spends the token
    budget on chain-of-thought and can return empty content. We disable thinking and
    give a generous max_tokens so the structured answer always comes back.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content, None
            last_err = "empty response from model"
        except Exception as e:
            last_err = str(e)
            # 1113 = insufficient balance: retrying won't help, fail fast.
            if "1113" in last_err:
                return None, last_err
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # 1s, 2s backoff
    return None, last_err


def parse_response(response_text):
    """Parse GLM's structured response."""
    result = {"prediction": "", "confidence": "", "reasoning": ""}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("PREDICTION:"):
            pred = line.split(":", 1)[1].strip().lower()
            # Normalize various possible responses
            if pred in ["suicidal", "yes", "true", "positive"]:
                result["prediction"] = "suicidal"
            elif pred in ["not_suicidal", "not suicidal", "no", "false", "negative"]:
                result["prediction"] = "not_suicidal"
            else:
                result["prediction"] = pred
        elif line.startswith("CONFIDENCE:"):
            conf = line.split(":", 1)[1].strip().lower()
            if conf in ["high", "very high"]:
                result["confidence"] = "high"
            elif conf in ["low", "very low"]:
                result["confidence"] = "low"
            else:
                result["confidence"] = "medium"
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
    output_path = _data_dir() / "umd_annotation_sheet_glm.xlsx"

    # Load Excel
    print(f"Loading {input_path}...")
    df = pd.read_excel(input_path, engine='openpyxl')

    # Count samples to annotate
    to_annotate = df[df["llm_prediction"].isna() | (df["llm_prediction"] == "")]
    print(f"Samples to annotate: {len(to_annotate)}")

    if len(to_annotate) == 0:
        print("All samples already annotated!")
        # Still show summary
        annotated = df[df["llm_prediction"] != ""]
        if len(annotated) > 0:
            accuracy = (annotated["llm_correct"] == "YES").sum() / len(annotated) * 100
            print(f"\nCurrent accuracy: {accuracy:.1f}%")
        return

    # Get API key
    api_key = API_KEY or os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("Error: Set ZHIPUAI_API_KEY environment variable or API_KEY in script")
        print("Get your key at: https://open.bigmodel.cn/")
        print("\nTo set the key:")
        print("  export ZHIPUAI_API_KEY='your-key-here'  # Linux/Mac")
        print("  set ZHIPUAI_API_KEY=your-key-here       # Windows")
        return

    # Initialize GLM client and pick the best model this account can call
    client = ZhipuAI(api_key=api_key)
    print("Selecting model (best available)...")
    model = pick_model(client)
    print(f"Using model: {model}")

    # Annotate each sample
    for idx, row in to_annotate.iterrows():
        print(f"\n[{len(df) - len(to_annotate) + idx + 1}/{len(df)}] Annotating sample_{row['sample_id']}...")

        # Truncate text for API
        text = row["text"][:3000] if len(row["text"]) > 3000 else row["text"]
        prompt = PROMPT_TEMPLATE.format(text=text)

        response_text, error = call_glm(client, prompt, model)

        if error:
            print(f"  API Error: {error}")
            df.at[idx, "llm_prediction"] = "ERROR"
            df.at[idx, "llm_reasoning"] = error
            continue

        result = parse_response(response_text)

        df.at[idx, "llm_prediction"] = result["prediction"]
        df.at[idx, "llm_confidence"] = result["confidence"]
        df.at[idx, "llm_reasoning"] = result["reasoning"]

        # Check correctness
        if result["prediction"]:
            is_correct = "YES" if result["prediction"] == row["binary_label"] else "NO"
            df.at[idx, "llm_correct"] = is_correct
            print(f"  Prediction: {result['prediction']} | Truth: {row['binary_label']} | {is_correct}")
        else:
            print(f"  Warning: Could not parse prediction from: {response_text[:100]}")

        # Save progress every 5 samples
        if (idx + 1) % 5 == 0:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"  [Progress saved]")

    # Final save
    df.to_excel(output_path, index=False, engine='openpyxl')

    # Print summary
    print("\n" + "="*50)
    print("GLM ANNOTATION SUMMARY")
    print("="*50)

    annotated = df[df["llm_prediction"].isin(["suicidal", "not_suicidal"])]
    if len(annotated) > 0:
        accuracy = (annotated["llm_correct"] == "YES").sum() / len(annotated) * 100

        print(f"Total annotated: {len(annotated)}")
        print(f"Accuracy: {accuracy:.1f}%")

        print("\nConfusion Matrix:")
        confusion = pd.crosstab(annotated["binary_label"], annotated["llm_prediction"],
                               rownames=["Truth"], colnames=["Prediction"])
        print(confusion)

        # Per-class metrics
        if "suicidal" in confusion.index:
            suicidal_correct = confusion.loc["suicidal", "suicidal"] if "suicidal" in confusion.columns else 0
            suicidal_total = confusion.loc["suicidal"].sum()
            print(f"\nSuicidal recall: {suicidal_correct}/{suicidal_total} ({suicidal_correct/suicidal_total*100:.1f}%)")

        if "not_suicidal" in confusion.index:
            not_suicidal_correct = confusion.loc["not_suicidal", "not_suicidal"] if "not_suicidal" in confusion.columns else 0
            not_suicidal_total = confusion.loc["not_suicidal"].sum()
            print(f"Not suicidal recall: {not_suicidal_correct}/{not_suicidal_total} ({not_suicidal_correct/not_suicidal_total*100:.1f}%)")

    else:
        print("No valid annotations found.")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
