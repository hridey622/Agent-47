import json
import re

def extract_json_from_model_output(model_output: str) -> dict:
    """Extracts a JSON object from a model output string.

    Args:
        model_output: The model output string.

    Returns:
        The extracted JSON object as a dict.

    Raises:
        ValueError: If no JSON object is found or if the JSON is malformed.
    """
    # Greedy match to find content between the first ```json and the last ```
    match = re.search(r"```json\n(.*?)\n```", model_output, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: try to find JSON directly if no markdown code block
        # This looks for the first '{' and the last '}'
        first_brace = model_output.find('{')
        last_brace = model_output.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = model_output[first_brace : last_brace + 1]
        else:
            raise ValueError("No JSON object found in model output")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}. JSON string was: {json_str}") from e 