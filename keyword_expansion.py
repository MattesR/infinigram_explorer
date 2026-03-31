"""
Keyword Expansion Pipeline
For a given query, returns a dict where:
  - keys   = semantic groups identified in the query
  - values = list of keyword expansions for that group
"""

import json
import os
import re


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a search query expansion expert.

Given a user query, identify the distinct semantic groups in the query and for each group generate keyword expansions: synonyms, related terms, broader/narrower concepts, and common variations.

Respond ONLY with a valid JSON object. No markdown fences, no explanation.

The output format is:
{
  "<semantic_group>": ["keyword1", "keyword2", ...],
  ...
}

Rules:
- Use concise, lowercase snake_case for group keys (e.g. "location", "job_role", "time_period")
- Each group should have 5–10 keyword expansions
- Expansions should cover: synonyms, abbreviations, related terms, and common variants
- Do not repeat the original query words as-is — expand beyond them
- If the query is very short (1–2 words), return 2–3 semantic groups by inferring intent

Example:
Query: "cheap flights to Paris in summer"
Output:
{
  "price": ["budget", "low-cost", "affordable", "discount", "deal", "economy", "inexpensive", "cheap fare"],
  "transport": ["flight", "airfare", "airline", "plane ticket", "direct flight", "connecting flight", "round trip"],
  "destination": ["Paris", "France", "CDG", "Orly", "Île-de-France", "French capital", "Europe"],
  "time_period": ["summer", "June", "July", "August", "peak season", "school holidays", "warm months"]
}"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def call_claude(query: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024) -> dict:
    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    return parse_json_response(message.content[0].text)


def call_openai(query: str, model: str = "gpt-4o", max_tokens: int = 1024) -> dict:
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return parse_json_response(response.choices[0].message.content)


def call_ollama(query: str, model: str = "llama3.2", host: str = "http://localhost:11434") -> dict:
    import requests
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        },
    )
    response.raise_for_status()
    return parse_json_response(response.json()["message"]["content"])


PROVIDERS = {
    "claude": call_claude,
    "openai": call_openai,
    "ollama": call_ollama,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def expand_query(query: str, provider: str = "claude") -> dict[str, list[str]]:
    """
    Expand a query into semantic groups with keyword expansions.

    Returns:
        {
            "semantic_group": ["keyword1", "keyword2", ...],
            ...
        }
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

    result = PROVIDERS[provider](query)

    # Validate: must be a flat dict of str -> list[str]
    if not isinstance(result, dict):
        raise ValueError(f"Expected a dict, got {type(result)}")
    for k, v in result.items():
        if not isinstance(v, list):
            raise ValueError(f"Expected list for group '{k}', got {type(v)}")

    return result


def expand_queries(queries: list[str], provider: str = "claude") -> dict[str, dict[str, list[str]]]:
    """Expand multiple queries. Returns {query: expansion_dict}."""
    return {q: expand_query(q, provider=provider) for q in queries}


# ---------------------------------------------------------------------------
# Prompt evaluation
# ---------------------------------------------------------------------------

def evaluate_expansion(result: dict, checks: dict) -> dict:
    """
    checks is a dict of group_name -> minimum number of expansions expected.
    Example: {"price": 3, "destination": 3}
    """
    issues = []

    for group, min_count in checks.items():
        if group not in result:
            issues.append(f"Missing expected group: '{group}'")
        elif len(result[group]) < min_count:
            issues.append(f"Group '{group}' has {len(result[group])} keywords, expected >= {min_count}")

    total_keywords = sum(len(v) for v in result.values())
    if total_keywords < 10:
        issues.append(f"Very few total keywords ({total_keywords}), expansion may be too shallow")

    return {"valid": len(issues) == 0, "issues": issues, "total_keywords": total_keywords, "groups": list(result.keys())}


def run_eval_suite(test_cases: list[dict], provider: str = "claude") -> None:
    """
    Each test case:
      - "query": str
      - "expected_groups": list[str]   minimum groups you want to see
      - "min_keywords_per_group": int  (optional, default 4)
    """
    print(f"\n=== Keyword Expansion Eval ({provider}) ===\n")
    passed = 0

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected_groups = case.get("expected_groups", [])
        min_kw = case.get("min_keywords_per_group", 4)

        print(f"Test {i+1}: '{query}'")
        try:
            result = expand_query(query, provider=provider)
            checks = {g: min_kw for g in expected_groups}
            eval_result = evaluate_expansion(result, checks)

            if eval_result["valid"]:
                print(f"  PASS — {eval_result['groups']} ({eval_result['total_keywords']} keywords total)")
                passed += 1
            else:
                print(f"  FAIL — {eval_result['issues']}")
                print(f"  Got groups: {eval_result['groups']}")
        except Exception as e:
            print(f"  ERROR — {e}")

    print(f"\nResult: {passed}/{len(test_cases)} passed\n")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Single query expansion
    query = "machine learning engineer jobs in Berlin"
    print(f"Query: {query}\n")

    result = expand_query(query, provider="claude")
    print(json.dumps(result, indent=2))

    # Evaluation suite
    test_cases = [
        {
            "query": "cheap flights to Paris in summer",
            "expected_groups": ["price", "transport", "destination", "time_period"],
            "min_keywords_per_group": 4,
        },
        {
            "query": "remote python developer jobs",
            "expected_groups": ["work_arrangement", "programming_language", "job_role"],
            "min_keywords_per_group": 4,
        },
        {
            "query": "best Italian restaurants near me",
            "expected_groups": ["cuisine", "establishment_type", "quality"],
            "min_keywords_per_group": 3,
        },
    ]

    run_eval_suite(test_cases, provider="claude")