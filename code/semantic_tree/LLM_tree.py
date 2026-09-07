import json
import time
from typing import List

from openai import OpenAI


# =========================
# Configuration
# =========================

BASE_URL = "https://your-api-base-url/v1"
API_KEY = "your-api-key"
MODEL_NAME = "gpt-4o-mini"

MAX_RETRIES = 3
TEMPERATURE = 0.2


# =========================
# Prompt
# =========================

SYSTEM_PROMPT = """
You are an expert in semantic decomposition for multi-hop table fact-checking.

Given a claim, convert it into a linearized semantic-tree format:
- Output a JSON array of strings.
- Each string is a semantic unit appearing as a contiguous span or a semantically faithful sub-span of the claim.
- The list must be ordered from shorter / more local units to longer / more compositional units.
- The last item must be the full original claim.
- The structure must support a unique tree under this rule:
  node a is a direct child of node b iff:
  1) a is contained in b, and
  2) there does not exist a node c such that a is contained in c and c is contained in b.
- Prefer meaningful semantic units such as entities, attributes, time constraints, quantities, and intermediate compositions.
- Do not include explanations.
- Do not include markdown.
- Do not output anything except the JSON array.
"""

USER_PROMPT_TEMPLATE = """
Claim:
{claim}

Return only a JSON array of strings.
"""


# =========================
# API Client
# =========================

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# =========================
# Validation Utilities
# =========================

def normalize_text(text: str) -> str:
    """Normalize whitespace for safer substring checks."""
    return " ".join(text.strip().split())


def is_sorted_by_length(items: List[str]) -> bool:
    """Check whether the list is sorted by non-decreasing string length."""
    lengths = [len(x) for x in items]
    return all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1))


def validate_semantic_tree_list(claim: str, items: List[str]) -> None:
    """
    Validate the returned semantic-tree list.

    Rules enforced:
    - Must be a non-empty list of strings.
    - Must be sorted from short to long.
    - The last element must exactly equal the original claim after normalization.
    - Every non-root node must be contained in at least one later node.
    """
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("Output must be a non-empty list.")

    if not all(isinstance(x, str) and x.strip() for x in items):
        raise ValueError("All items must be non-empty strings.")

    norm_items = [normalize_text(x) for x in items]
    norm_claim = normalize_text(claim)

    if not is_sorted_by_length(norm_items):
        raise ValueError("Items are not sorted from short to long.")

    if norm_items[-1] != norm_claim:
        raise ValueError("The last item must be the full original claim.")

    for i in range(len(norm_items) - 1):
        a = norm_items[i]
        found_parent = False
        for j in range(i + 1, len(norm_items)):
            b = norm_items[j]
            if a in b:
                found_parent = True
                break
        if not found_parent:
            raise ValueError(f"Node has no parent container: {items[i]}")


# =========================
# Core Function
# =========================

def claim_to_semantic_tree_list(claim: str) -> List[str]:
    """
    Convert a claim into the linearized semantic-tree format.

    Returns:
        A list of strings sorted from short to long,
        where the last element is the full claim.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(claim=claim)
                    },
                ],
            )

            content = response.choices[0].message.content.strip()
            items = json.loads(content)

            validate_semantic_tree_list(claim, items)
            return items

        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Failed: {e}")
            time.sleep(1)

    raise RuntimeError(f"Failed to generate a valid semantic tree list. Last error: {last_error}")


# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    claim = "The Norman conquerors established a historical site in England that attracted 2,389,548 tourists in 2009."

    tree_list = claim_to_semantic_tree_list(claim)

    print(json.dumps(tree_list, ensure_ascii=False, indent=2))