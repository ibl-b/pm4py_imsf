from enum import Enum
from pm4py.util import exec_utils
from typing import Optional, Dict, Any
import base64, os
from pm4py.util import constants


class Parameters(Enum):
    API_URL = "api_url"
    API_KEY = "api_key"
    ANTHROPIC_MODEL = "anthropic_model"
    IMAGE_PATH = "image_path"
    MAX_TOKENS = "max_tokens"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def apply(prompt: str, parameters: Optional[Dict[Any, Any]] = None) -> str:
    import requests

    if parameters is None:
        parameters = {}

    image_path = exec_utils.get_param_value(Parameters.IMAGE_PATH, parameters, None)
    api_key = exec_utils.get_param_value(Parameters.API_KEY, parameters, constants.ANTHROPIC_API_KEY)
    api_url = exec_utils.get_param_value(Parameters.API_URL, parameters, None)
    max_tokens = exec_utils.get_param_value(Parameters.MAX_TOKENS, parameters, 8192)
    simple_content_specification = image_path is None

    if api_url is None:
        api_url = "https://api.anthropic.com/v1/"
    else:
        if not api_url.endswith("/"):
            api_url += "/"

    model = exec_utils.get_param_value(Parameters.ANTHROPIC_MODEL, parameters, constants.ANTHROPIC_DEFAULT_MODEL)

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key
    }

    messages = []
    if simple_content_specification:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    payload = {
        "model": model,
        "max_tokens": max_tokens
    }

    if image_path is not None:
        image_format = os.path.splitext(image_path)[1][1:].lower()
        base64_image = encode_image(image_path)
        artefact = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/" + image_format,
                "data": base64_image
            }
        }
        messages[0]["content"].append(artefact)

    payload["messages"] = messages

    response = requests.post(api_url+"messages", headers=headers, json=payload).json()

    if "error" in response:
        # raise an exception when the request fails, with the provided message
        raise Exception(response["error"]["message"])

    return response["content"][0]["text"]
