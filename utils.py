import json
import re
from typing import Any


def extract_json_from_response(content: str) -> str:
    """Extract JSON from markdown code blocks or plain text."""
    # Try to find JSON in code blocks first
    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        return json_match.group(1).strip()

    # If no code blocks, try to find JSON-like content
    # Look for content that starts with { and ends with }
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # Return original content if no patterns found
    return content.strip()


def count_nested_objects_and_depth(
    obj: dict[str, Any], depth: int = 0
) -> tuple[int, int]:
    """Count nested objects and track maximum depth."""
    if not isinstance(obj, dict):
        return 0, depth

    count = 1
    max_depth = depth

    for value in obj.values():
        if isinstance(value, dict):
            nested_count, nested_depth = count_nested_objects_and_depth(
                value, depth + 1
            )
            count += nested_count
            max_depth = max(max_depth, nested_depth)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            nested_count, nested_depth = count_nested_objects_and_depth(
                value[0], depth + 1
            )
            count += nested_count
            max_depth = max(max_depth, nested_depth)

    return count, max_depth


def count_literal_fields(obj: dict[str, Any]) -> int:
    """Count literal fields (strings, numbers, booleans, enums)."""
    if not isinstance(obj, dict):
        return 0

    literal_keys = {"type", "enum", "const"}
    count = sum(1 for key in obj.keys() if key in literal_keys)

    for value in obj.values():
        if isinstance(value, dict):
            count += count_literal_fields(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            count += count_literal_fields(value[0])

    return count


def estimate_complexity(schema: dict[str, Any]) -> dict[str, int]:
    """Estimate schema complexity for adaptive processing."""
    nested_objects, max_depth = count_nested_objects_and_depth(schema)
    literals = count_literal_fields(schema)

    return {
        "nested_objects": nested_objects,
        "max_depth": max_depth,
        "literals": literals,
        "total_keys": len(json.dumps(schema)),
    }


def build_adaptive_prompt(
    schema: dict[str, Any], text: str, complexity: dict[str, int]
) -> list[dict[str, str]]:
    """Build adaptive prompt based on schema complexity."""

    # Base system prompt
    system_prompt = """You are a highly accurate JSON extractor. Your task is to extract structured data from unstructured text according to a provided JSON schema.

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON that strictly conforms to the schema
2. Do NOT include markdown code blocks or any other formatting
3. If a field cannot be determined from the text, use null or appropriate default values
4. Ensure all required fields are present
5. Pay attention to field types (string, number, boolean, array, object)
6. For enum fields, only use values specified in the schema"""

    # Adjust strategy based on complexity
    if complexity["nested_objects"] > 50 or complexity["max_depth"] > 5:
        system_prompt += "\n\nThis is a complex schema with deep nesting. Process methodically, section by section."

    if complexity["literals"] > 100:
        system_prompt += "\n\nThis schema has many literal fields. Be precise with data types and enum values."

    # User prompt with schema and text
    user_prompt = f"""JSON Schema:
{json.dumps(schema, indent=2)}

Text to extract from:
{text}

Extract the data as JSON:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_model_for_complexity(complexity: dict[str, int], text_length: int) -> str:
    """Select appropriate model based on complexity and input size."""
    # For very large inputs or complex schemas, use more capable models
    if text_length > 100000 or complexity["nested_objects"] > 100:
        return "gpt-4o"  # Most capable for complex tasks
    elif complexity["nested_objects"] > 30 or text_length > 50000:
        return "gpt-4o-mini"  # Good balance
    else:
        return "gpt-4o-mini"  # Efficient for simpler tasks
