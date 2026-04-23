from datetime import datetime
from fastapi import Body, FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any, Dict, List

import cv2
import json
import shutil
import tempfile
import uuid

# TODO: Add dashboard auto-refresh or polling for when video / extract frames flow concluded

# Directories for datasets
DATASET_DIR = "dataset"
DATASET_SQUIRREL_DIR = Path(DATASET_DIR) / "squirrel"
DATASET_NOT_DIR = Path(DATASET_DIR) / "not_squirrel"

# Directory for image storage
IMAGE_UPLOADS_DIR = "image_uploads"
Path(IMAGE_UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

METADATA_PATH = Path(IMAGE_UPLOADS_DIR) / "metadata.json"

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGE_UPLOADS_DIR), name="images")


#######################################################################
######################### ENDPOINTS ###################################
#######################################################################


# Simple server life check
@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(filter: str = "all"):
    # Load records (from metadata.json)
    records = load_metadata()
    records.sort(key=lambda r: r.get("modified_epoch", 0), reverse=True)

    # Optional filtering
    if filter == "unlabeled":
        records = [r for r in records if not r.get("label")]
    elif filter == "squirrel":
        records = [r for r in records if r.get("label") == "squirrel"]
    elif filter == "not_squirrel":
        records = [r for r in records if r.get("label") == "not_squirrel"]
    else:
        filter = "all"

    # Limit for page performance
    records = records[:48]

    script = """
    <script>
    async function setLabel(recordId, label) {
      const res = await fetch('/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record_id: recordId, label: label })
      });
      const data = await res.json();
      const el = document.getElementById('label-' + recordId);

      if (data.status === 'ok') {
        el.textContent = 'Label: ' + data.label;
      } else {
        el.textContent = 'Error: ' + (data.message || 'unknown');
      }
    }
    </script>
    """

    def label_badge(label: str | None) -> str:
        if label == "squirrel":
            return "🐿️ squirrel"
        if label == "not_squirrel":
            return "🚫 not_squirrel"
        return "❓ unlabeled"

    cards = "\n".join(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; margin:8px; width:180px;">
          <a href="{r.get('url')}" target="_blank">
            <img src="{r.get('url')}" style="height:140px; border-radius:10px;" />
          </a>

          <div id="label-{r.get('id')}" style="font-size:12px; margin-top:6px; opacity:0.9;">
            {label_badge(r.get("label"))}
          </div>

          <div style="margin-top:6px; display:flex; gap:6px;">
            <button onclick="setLabel('{r.get('id')}', 'squirrel')">🐿️</button>
            <button onclick="setLabel('{r.get('id')}', 'not_squirrel')">🚫</button>
          </div>

          <div style="font-size:10px; margin-top:6px; opacity:0.6; text-align:center;">
            {r.get("saved_as")}
          </div>
        </div>
        """
        for r in records
    )

    html = f"""
    <html>
      <head>
        <title>Squirrel AI Dashboard</title>
        {script}
      </head>
      <body style="font-family: Arial; padding: 20px;">
        <h1>🐿️ Squirrel AI Dashboard</h1>

        <div style="margin: 12px 0;">
          <strong>Filter:</strong>
          <a href="/dashboard?filter=all">All</a> |
          <a href="/dashboard?filter=unlabeled">Unlabeled</a> |
          <a href="/dashboard?filter=squirrel">Squirrel</a> |
          <a href="/dashboard?filter=not_squirrel">Not Squirrel</a>
        </div>

        <p>Showing {len(records)} records (newest first)</p>

        <div style="display:flex; flex-wrap:wrap;">
          {cards}
        </div>
      </body>
    </html>
    """
    return html


@app.post("/export_dataset")
def export_dataset(overwrite: bool = False):
    DATASET_SQUIRREL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_NOT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_metadata()

    exported = 0
    skipped = 0

    for r in records:
        label = r.get("label")
        saved_as = r.get("saved_as")

        if label not in {"squirrel", "not_squirrel"} or not saved_as:
            continue

        src = Path(IMAGE_UPLOADS_DIR) / saved_as
        if not src.exists():
            skipped += 1
            continue

        dst_dir = DATASET_SQUIRREL_DIR if label == "squirrel" else DATASET_NOT_DIR
        dst = dst_dir / saved_as

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        shutil.copy2(src, dst)
        exported += 1

    return {"status": "ok", "exported": exported, "skipped": skipped}


# Upload a video, extract frames with OpenCV, and save them as images + metadata
# Upload video → extract frames → save as images for labeling
@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    label: str | None = None,
    identity: str | None = None,
):
    if not file.filename:
        return {"status": "error", "message": "No filename provided"}

    if not file.content_type or not file.content_type.startswith("video/"):
        return {
            "status": "error",
            "message": f"Expected a video upload, got: {file.content_type}",
        }

    if label is not None and label not in {"squirrel", "not_squirrel"}:
        return {
            "status": "error",
            "message": "Expected a binary of either squirrel or not squirrel",
        }

    if identity is not None and identity not in {"roughneck"}:
        return {
            "status": "error",
            "message": "Invalid video identity",
        }

    suffix = Path(file.filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, suffix=str(suffix)
    ) as tmp:
        temp_video_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        created_records = extract_frames_from_video(
            video_path=temp_video_path,
            every_n_seconds=every_n_seconds,
            max_frames=max_frames,
            label=label,
            identity=identity,
        )

        return {
            "status": "success",
            "original_video_filename": file.filename,
            "content_type": file.content_type,
            "frames_created": len(created_records),
            "every_n_seconds": every_n_seconds,
            "max_frames": max_frames,
            "records": created_records[:10],
        }

    finally:
        temp_path = Path(temp_video_path)
        if temp_path.exists():
            temp_path.unlink()


@app.get("/gallery")
def gallery(limit: int = 20):
    folder = Path(IMAGE_UPLOADS_DIR)
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    items = [
        {
            "filename": p.name,
            "url": f"/images/{p.name}",
            "modified_epoch": p.stat().st_mtime,
        }
        for p in files[:limit]
    ]

    return {"status": "ok", "count": len(items), "items": items}


@app.post("/label")
def label_image(
    record_id: str = Body(...),
    label: str = Body(...),
):
    if label not in {"squirrel", "not_squirrel"}:
        return {
            "status": "error",
            "message": "label must be 'squirrel' or 'not_squirrel'",
        }

    ok = set_label(record_id, label)
    if not ok:
        return {"status": "error", "message": f"record_id not found: {record_id}"}

    return {"status": "ok", "record_id": record_id, "label": label}


@app.get("/latest")
def latest():
    folder = Path(IMAGE_UPLOADS_DIR)
    files = [p for p in folder.iterdir() if p.is_file()]

    if not files:
        return {"response_status": "error", "message": "No images uploaded yet"}

    newest = max(files, key=lambda p: p.stat().st_mtime)

    return {
        "response_status": "ok",
        "filename": newest.name,
        "url": f"/images/{newest.name}",
        "modified_epoch": newest.stat().st_mtime,
    }


@app.get("/records")
def records(limit: int = 50):
    data = load_metadata()
    data.sort(key=lambda r: r.get("modified_epoch", 0), reverse=True)
    return {"status": "ok", "count": len(data[:limit]), "items": data[:limit]}


@app.get("/stats")
def stats():
    records = load_metadata()
    total = len(records)
    squirrel = sum(1 for r in records if r.get("label") == "squirrel")
    not_squirrel = sum(1 for r in records if r.get("label") == "not_squirrel")
    labeled = squirrel + not_squirrel
    unlabeled = total - labeled

    return {
        "status": "ok",
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "squirrel": squirrel,
        "not_squirrel": not_squirrel,
    }


# Upload images, save them to the image_uploads folder for model comparisons
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        return {"response_status": "error", "error_msg": "No filename provided"}

    saved_name = make_unique_filename(file.filename)
    file_path = Path(IMAGE_UPLOADS_DIR) / saved_name

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    stat = file_path.stat()

    record = {
        "id": saved_name,  # use filename as id for now
        "original_filename": file.filename,
        "saved_as": saved_name,
        "content_type": file.content_type,
        "size_bytes": stat.st_size,
        "modified_epoch": stat.st_mtime,
        "url": f"/images/{saved_name}",
        "label": None,  # later: "squirrel" / "not_squirrel"
        "prediction": None,  # later: {"squirrel": 0.92}
        "identity": None,
    }
    add_record(record)

    return {"status": "success", **record}


################################################
############## HELPER FUNCTIONS ################
################################################


def add_record(record: Dict[str, Any]) -> None:
    records = load_metadata()
    records.append(record)
    save_metadata(records)


def load_metadata() -> List[Dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        # If file is corrupted/empty, start fresh (safe for now)
        return []


# Helper method for unique filename creation
def make_unique_filename(original: str) -> str:
    """
    Returns: YYYYMMDD_HHMMSS_mmm_uuid8.ext
    Example: 20260304_103355_214_a1b2c3d4.png
    """
    ext = Path(original).suffix.lower()  # includes the dot, e.g. ".png"
    if not ext:
        ext = ".bin"

    # Timestamp (year-month-day_hour-min-sec_microsec) trimmed for readability
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]

    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}{ext}"


def save_metadata(records: List[Dict[str, Any]]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def set_label(record_id: str, label: str) -> bool:
    records = load_metadata()
    updated = False

    for r in records:
        if r.get("id") == record_id:
            r["label"] = label
            updated = True
            break

    if updated:
        save_metadata(records)

    return updated


def extract_frames_from_video(
    video_path: str,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    label: str | None = None,
    identity: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Extract frames from a video file, save them into IMAGE_UPLOADS_DIR,
    create metadata records, and return the created records.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS")

    frame_interval = max(int(source_fps * every_n_seconds), 1)

    frame_index = 0
    saved_count = 0
    created_records: List[Dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            saved_name = make_unique_filename(f"video_frame_{saved_count}.jpg")
            file_path = Path(IMAGE_UPLOADS_DIR) / saved_name

            success = cv2.imwrite(str(file_path), frame)
            if not success:
                frame_index += 1
                continue

            stat = file_path.stat()

            record = {
                "id": saved_name,
                "original_filename": f"video_frame_{saved_count}.jpg",
                "saved_as": saved_name,
                "content_type": "image/jpeg",
                "size_bytes": stat.st_size,
                "modified_epoch": stat.st_mtime,
                "url": f"/images/{saved_name}",
                "label": label,
                "prediction": None,
                "source": "video_upload",
                "identity": identity,
            }

            add_record(record)
            created_records.append(record)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_index += 1

    cap.release()
    return created_records
