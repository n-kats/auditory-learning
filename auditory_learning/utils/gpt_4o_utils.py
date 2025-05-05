import base64
from io import BytesIO

from PIL import Image


def to_image_content(image: Image, image_type: str):
    with BytesIO() as f_out:
        image.save(f_out, format=image_type)
        encoded = base64.b64encode(f_out.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
    }


def run_gpt_4o(client, messages, model="gpt-4o", json_mode=False, **kwargs):
    if json_mode:
        json_object = {"type": "json_object"}
        assert kwargs.get("response_format", json_object) == json_object
        kwargs["response_format"] = json_object

    return client.chat.completions.create(model=model, messages=messages, **kwargs).choices[0].message.content
