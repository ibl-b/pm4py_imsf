from enum import Enum
from pm4py.util import exec_utils
from typing import Optional, Dict, Any
import base64, os
from pm4py.util import constants


class Parameters(Enum):
    API_KEY = "api_key"
    GOOGLE_MODEL = "google_model"
    IMAGE_PATH = "image_path"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def apply(prompt: str, parameters: Optional[Dict[Any, Any]] = None) -> str:
    import requests

    if parameters is None:
        parameters = {}

    image_path = exec_utils.get_param_value(Parameters.IMAGE_PATH, parameters, None)
    api_key = exec_utils.get_param_value(Parameters.API_KEY, parameters, constants.GOOGLE_API_KEY)
    model = exec_utils.get_param_value(Parameters.GOOGLE_MODEL, parameters, constants.GOOGLE_DEFAULT_MODEL)

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {"parts": [
                {"text": prompt}
            ]}
        ]
    }

    if image_path is not None:
        image_format = os.path.splitext(image_path)[1][1:].lower()
        base64_image = encode_image(image_path)
        spec = {"inline_data": {
            "mime_type": "image/" + image_format,
            "data": base64_image
        }
        }
        payload["contents"][0]["parts"].append(spec)

    url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + api_key

    response = requests.post(url, headers=headers, json=payload).json()

    if "error" in response:
        # raise an exception when the request fails, with the provided message
        raise Exception(response["error"]["message"])

    return response["candidates"][0]["content"]["parts"][0]["text"]
