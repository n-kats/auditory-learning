import base64
from dataclasses import dataclass
from io import BytesIO

from PIL import Image


@dataclass(frozen=True)
class GptResult:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def to_image_content(image: Image, image_type: str):
    with BytesIO() as f_out:
        image.save(f_out, format=image_type)
        encoded = base64.b64encode(f_out.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
    }


def run_gpt(client, messages, model="gpt-4o", json_mode=False, reasoning_effort=None, **kwargs):
    if json_mode:
        json_object = {"type": "json_object"}
        assert kwargs.get("response_format", json_object) == json_object
        kwargs["response_format"] = json_object
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    return GptResult(
        content=response.choices[0].message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
