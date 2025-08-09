import json
import os
import sys
import threading
from pathlib import Path
from queue import Queue
from uuid import uuid4

import fastapi
import httpx
import openai
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel

from auditory_learning.utils.gpt_4o_utils import run_gpt_4o, to_image_content
from auditory_learning.utils.voice_utils import VoiceVoxSpeaker, text_to_wav

data_dir = Path("_data")

load_dotenv()
app = fastapi.FastAPI()
client = openai.Client()
url_to_request_id_path = data_dir / "url_to_request_id.json"
if url_to_request_id_path.exists():
    url_to_request_id = json.loads(url_to_request_id_path.read_text())
else:
    url_to_request_id = {}
    url_to_request_id_path.parent.mkdir(parents=True, exist_ok=True)
    url_to_request_id_path.write_text(json.dumps(url_to_request_id))

frontend_dir = (Path(__file__).parent.parent / "frontend").resolve()
app.mount("/static", StaticFiles(directory=frontend_dir / "dist"), name="static")
app.mount("/assets", StaticFiles(directory=frontend_dir / "dist/assets/"))
prompt_path = Path(os.environ.get("AUDITORY_LEARNING_PROMPT_PATH", "prompt.txt"))


@app.get("/")
def root():
    return fastapi.responses.RedirectResponse("/static/index.html")


class InitRequest(BaseModel):
    url: str


class InitResponse(BaseModel):
    request_id: str
    page_num: int


@app.post("/init/")
def init(req: InitRequest) -> InitResponse:
    if req.url in url_to_request_id:
        request_id = url_to_request_id[req.url]
    else:
        request_id = str(uuid4())
        url_to_request_id[req.url] = request_id
        url_to_request_id_path.parent.mkdir(parents=True, exist_ok=True)
        url_to_request_id_path.write_text(json.dumps(url_to_request_id))
    work_dir = data_dir / request_id
    image_dir = work_dir / "images"
    pdf_path = work_dir / "pdf.pdf"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        print(f"[INFO] Download PDF from {req.url}", file=sys.stderr)
        pdf_path.write_bytes(httpx.get(req.url).content)
    pages = convert_from_path(pdf_path)
    for i, page in enumerate(pages, start=1):
        if not (image_dir / f"{i:04d}.png").exists():
            page.save(image_dir / f"{i:04d}.png")

    return InitResponse(request_id=request_id, page_num=len(pages))


class ImageRequest(BaseModel):
    request_id: str
    page: int


@app.post("/image/")
def image(req: ImageRequest) -> fastapi.responses.FileResponse:
    # 画像を返す
    work_dir = data_dir / req.request_id
    image_path = work_dir / "images" / f"{req.page:04d}.png"
    return fastapi.responses.FileResponse(image_path)


class ExplainRequest(BaseModel):
    request_id: str
    page: int


class ExplainResponse(BaseModel):
    explanation: str


speaker = VoiceVoxSpeaker(
    speaker_id="1",
    speed=1.5,
    volume=4,
    url="http://localhost:50021",
)


@app.post("/explain/")
def explain(req: ExplainRequest) -> ExplainResponse:
    image_path = data_dir / req.request_id / "images" / f"{req.page:04d}.png"
    cache_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"

    if image_path.exists() and cache_path.exists() and audio_path.exists():
        return ExplainResponse(explanation=cache_path.read_text())

    explanation = generate_explanation_through_queue(
        f"{req.request_id}:{req.page:04d}",
        (image_path, cache_path, audio_path),
    )

    next_image_path = data_dir / req.request_id / "images" / f"{req.page + 1:04d}.png"
    if next_image_path.exists():
        next_cache_path = data_dir / req.request_id / f"explain_{req.page + 1:04d}.txt"
        next_audio_path = data_dir / req.request_id / f"explain_{req.page + 1:04d}.mp3"
        if not (next_cache_path.exists() and next_audio_path.exists()):
            _ = reserve_generation(
                f"{req.request_id}:{req.page + 1:04d}",
                (next_image_path, next_cache_path, next_audio_path),
            )

    return ExplainResponse(explanation=explanation)


def generate_explanation(image_path):
    image = Image.open(image_path)
    image_type = "png"
    image_content = to_image_content(image, image_type)
    prompt = prompt_path.read_text().strip()
    response = run_gpt_4o(
        client,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    image_content,
                ],
            }
        ],
        json_mode=False,
        model="gpt-5-mini",
    )
    return response


generation_queue: Queue[tuple[str, tuple[Path, Path, Path]]] = Queue()
request_queues: dict[str, list[Queue]] = {}
request_queues_lock = threading.Lock()


def worker(fn, input_queue: Queue, output_queues: dict[str, list[Queue]]):
    while True:
        key, input_ = input_queue.get()
        if key not in output_queues:
            continue

        result = fn(key, *input_)
        if key in output_queues:
            with request_queues_lock:
                for output_queue in output_queues[key]:
                    output_queue.put(result)
                del output_queues[key]
        else:
            print(f"[ERROR] No output queue for key: {key}", file=sys.stderr)


def generation_task(task_id: str, image_path: Path, cache_path: Path, audio_path: Path) -> str:
    print(f"[INFO] Generating explanation for {image_path}", file=sys.stderr)
    explanation = generate_explanation(image_path)
    cache_path.write_text(explanation)
    text_to_wav(explanation, speaker, audio_path)
    print(f"[INFO] Finished generating explanation for {image_path}", file=sys.stderr)
    print(f"[INFO] Explanation saved to {cache_path}", file=sys.stderr)
    print(f"[INFO] Audio saved to {audio_path}", file=sys.stderr)
    return explanation


threading.Thread(target=worker, args=(generation_task, generation_queue, request_queues), daemon=True).start()


def reserve_generation(task_id: str, args: tuple[Path, Path, Path]) -> Queue[str]:
    queue = Queue()
    with request_queues_lock:
        if task_id not in request_queues:
            request_queues[task_id] = []
        request_queues[task_id].append(queue)
    generation_queue.put((task_id, args))
    return queue


def generate_explanation_through_queue(task_id: str, args) -> str:
    queue = reserve_generation(task_id, args)
    return queue.get()


@app.post("/audio/")
def audio(req: ExplainRequest) -> fastapi.responses.FileResponse:
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"
    if not audio_path.exists():
        explanation_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
        explanation = explanation_path.read_text()
        text_to_wav(explanation, speaker, audio_path)
    return fastapi.responses.FileResponse(audio_path)


@app.post("/regenerate/")
def regenerate(req: ExplainRequest) -> ExplainResponse:
    image_path = data_dir / req.request_id / "images" / f"{req.page:04d}.png"
    cache_path = data_dir / req.request_id / f"explain_{req.page:04d}.txt"
    audio_path = data_dir / req.request_id / f"explain_{req.page:04d}.mp3"
    explanation = generate_explanation_through_queue(
        f"{req.request_id}:{req.page:04d}",
        (image_path, cache_path, audio_path),
    )

    return ExplainResponse(explanation=explanation)
