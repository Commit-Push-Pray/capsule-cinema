"""OrcaRouter-hosted Kling video-generation adapter for capsule-cinema.

Routes video generation through api.orcarouter.ai which proxies Kling's API.
Supports text-to-video and image-to-video (first frame) generation modes.

Env vars:
    ORCAROUTER_API_KEY       API key (or "proxy-injected" when behind a proxy)
    ORCAROUTER_BASE_URL      Base URL (default: https://api.orcarouter.ai/v1)
    ORCAROUTER_VIDEO_MODEL   Override the default model
    ORCAROUTER_TIMEOUT       HTTP timeout in seconds (default: 60)
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Type

import requests
from pydantic import BaseModel, Field

from custom_tools.audio_generation.base_tool_compat import BaseTool
from .output_dir_utils import default_video_output_dir, resolve_video_output_dir


DEFAULT_ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
DEFAULT_MODEL = "kling/kling-v2-5-turbo"
DEFAULT_OUTPUT_DIR = default_video_output_dir("orcarouter_kling")

SUPPORTED_RATIOS = {"16:9", "9:16", "1:1"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE_LIMIT = 20 * 1024 * 1024

_TERMINAL_FAILURE_STATES = {"failed", "failure", "error", "cancelled", "canceled", "expired"}
_SUCCEEDED_STATES = {"success", "succeeded", "completed"}
_RUNNING_STATES = {"in_progress", "running", "processing"}
_QUEUED_STATES = {"not_start", "submitted", "queued", "pending"}


class OrcaRouterVideoSchema(BaseModel):
    prompt: str = Field(..., description="Video generation prompt")
    generation_type: str = Field(
        "text_to_video",
        description="text_to_video or image_to_video",
    )
    output_dir: str = Field(DEFAULT_OUTPUT_DIR, description="Local output directory")
    output_path: Optional[str] = Field(None, description="Exact local output path")
    image_path: Optional[str] = Field(None, description="First-frame image for image_to_video")
    first_frame_path: Optional[str] = Field(None, description="Alias for image_path")
    start_image_path: Optional[str] = Field(None, description="Compatibility alias for image_path")
    aspect_ratio: str = Field("9:16", description="Video aspect ratio: 16:9, 9:16, or 1:1")
    duration: int = Field(5, description="Video duration in seconds (5 or 10)")
    model: Optional[str] = Field(None, description="Model override (default: kling/kling-v2-5-turbo)")
    poll_interval: int = Field(5, description="Task polling interval in seconds")
    max_wait: int = Field(600, description="Maximum task wait in seconds")


def _image_to_data_uri(path: str) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Image does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"Unsupported image format {suffix}; supported: {supported}")
    size = source.stat().st_size
    if size > IMAGE_SIZE_LIMIT:
        raise ValueError(f"Image exceeds {IMAGE_SIZE_LIMIT // (1024 * 1024)} MB limit")
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _duration_seconds(value: int | float | str) -> int:
    text = str(value).strip().lower().rstrip("s")
    numeric = float(text)
    result = int(numeric)
    if result not in (5, 10):
        raise ValueError("OrcaRouter Kling duration must be 5 or 10 seconds")
    return result


def build_orcarouter_payload(
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    generation_type: str = "text_to_video",
    image_path: Optional[str] = None,
    aspect_ratio: str = "9:16",
    duration: int | str = 5,
) -> dict[str, Any]:
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt must not be empty")
    if len(prompt) > 2500:
        raise ValueError("Prompt must not exceed 2500 characters")

    if generation_type not in ("text_to_video", "image_to_video"):
        raise ValueError(f"Unsupported generation_type: {generation_type}")
    if aspect_ratio not in SUPPORTED_RATIOS:
        raise ValueError(f"Unsupported aspect_ratio: {aspect_ratio}; supported: {', '.join(sorted(SUPPORTED_RATIOS))}")
    if generation_type == "image_to_video" and not image_path:
        raise ValueError("image_to_video requires image_path or first_frame_path")
    if generation_type == "text_to_video" and image_path:
        raise ValueError("text_to_video does not accept an image; use image_to_video")

    dur = _duration_seconds(duration)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "metadata": {
            "duration": str(dur),
            "aspect_ratio": aspect_ratio,
        },
    }
    if image_path:
        if image_path.startswith(("https://", "http://", "data:")):
            payload["image"] = image_path
        else:
            payload["image"] = _image_to_data_uri(image_path)
    return payload


def _extract_task_id(payload: dict[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return str(
        data.get("task_id")
        or data.get("id")
        or payload.get("task_id")
        or payload.get("id")
        or ""
    )


def _parse_poll(payload: dict[str, Any]) -> tuple[str, int | None, str, str]:
    """Returns (normalized_state, progress_pct, result_url, error_message)."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    raw = str(data.get("status", "")).strip().lower()

    if raw in _SUCCEEDED_STATES:
        state = "succeeded"
    elif raw in _RUNNING_STATES:
        state = "running"
    elif raw in _QUEUED_STATES:
        state = "queued"
    elif raw in _TERMINAL_FAILURE_STATES:
        state = "failed"
    else:
        state = "unknown"

    progress = None
    p = data.get("progress", "")
    if isinstance(p, (int, float)):
        progress = int(p)
    elif isinstance(p, str) and p.rstrip("%").isdigit():
        progress = int(p.rstrip("%"))

    url = (
        data.get("result_url")
        or (data.get("metadata") or {}).get("url")
        or data.get("url")
        or ""
    )

    error = (
        data.get("fail_reason")
        or (
            data["error"].get("message")
            if isinstance(data.get("error"), dict)
            else ""
        )
        or ""
    )
    error = re.sub(r"https?://\S+", "[redacted-url]", str(error))[:300]
    return state, progress, url, error


def _download(url: str, destination: Path, *, timeout: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                total += len(chunk)
    return total


def _error_detail(response: requests.Response) -> tuple[str, str]:
    try:
        payload = response.json()
    except ValueError:
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    error = payload.get("error")
    if isinstance(error, dict):
        message = re.sub(r"https?://\S+", "[redacted-url]", str(error.get("message") or ""))[:300]
        code = str(error.get("code") or "")
        return code, message
    return "", str(payload.get("message") or "")[:300]


class OrcaRouterVideoGeneratorTool(BaseTool):
    name: str = "OrcaRouter Kling video generator"
    description: str = "Generate Kling videos via OrcaRouter and download results locally."
    args_schema: Type[BaseModel] = OrcaRouterVideoSchema

    def _run(
        self,
        prompt: str,
        generation_type: str = "text_to_video",
        output_dir: str = DEFAULT_OUTPUT_DIR,
        output_path: Optional[str] = None,
        image_path: Optional[str] = None,
        first_frame_path: Optional[str] = None,
        start_image_path: Optional[str] = None,
        aspect_ratio: str = "9:16",
        duration: int | str = 5,
        model: Optional[str] = None,
        poll_interval: int = 5,
        max_wait: int = 600,
        **extra: Any,
    ) -> dict[str, Any]:
        api_key = os.getenv("ORCAROUTER_API_KEY") or os.getenv("GENAI_MEDIA_API_KEY")
        if not api_key:
            return {"success": False, "error": "Missing env var: ORCAROUTER_API_KEY or GENAI_MEDIA_API_KEY"}
        if extra:
            unsupported = sorted(k for k, v in extra.items() if v is not None)
            if unsupported:
                return {"success": False, "error": "OrcaRouter Kling does not support: " + ", ".join(unsupported)}

        resolved_model = model or os.getenv("ORCAROUTER_VIDEO_MODEL") or DEFAULT_MODEL
        base_url = (os.getenv("ORCAROUTER_BASE_URL") or DEFAULT_ORCAROUTER_BASE_URL).rstrip("/")
        timeout = max(1, int(os.getenv("ORCAROUTER_TIMEOUT", "60")))

        resolved_image = image_path or first_frame_path or start_image_path
        if generation_type == "image_to_video" and not resolved_image:
            return {"success": False, "error": "image_to_video requires image_path, first_frame_path, or start_image_path"}

        try:
            payload = build_orcarouter_payload(
                prompt=prompt,
                model=resolved_model,
                generation_type=generation_type,
                image_path=resolved_image,
                aspect_ratio=aspect_ratio,
                duration=duration,
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            created = requests.post(
                f"{base_url}/video/generations",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if created.status_code >= 400:
                code, message = _error_detail(created)
                suffix = f" ({code})" if code else ""
                return {
                    "success": False,
                    "error": f"OrcaRouter video request failed: HTTP {created.status_code}{suffix}",
                    "provider_error_code": code or None,
                    "provider_error_message": message or None,
                }

            task_id = _extract_task_id(created.json())
            if not task_id:
                return {"success": False, "error": "OrcaRouter response did not contain task_id"}

            started = time.monotonic()
            state = "queued"
            result_url = ""
            while time.monotonic() - started <= max(1, int(max_wait)):
                time.sleep(max(1, int(poll_interval)))
                polled = requests.get(
                    f"{base_url}/video/generations/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout,
                )
                if polled.status_code >= 400:
                    code, message = _error_detail(polled)
                    suffix = f" ({code})" if code else ""
                    return {
                        "success": False,
                        "error": f"OrcaRouter poll failed: HTTP {polled.status_code}{suffix}",
                        "provider_error_code": code or None,
                        "provider_error_message": message or None,
                        "task_id": task_id,
                    }
                state, _progress, result_url, error_msg = _parse_poll(polled.json())
                if state == "failed":
                    return {
                        "success": False,
                        "error": f"OrcaRouter task failed: {error_msg or 'unknown'}",
                        "task_id": task_id,
                    }
                if state == "succeeded" and result_url:
                    break

            if state != "succeeded" or not result_url:
                return {
                    "success": False,
                    "error": f"OrcaRouter task timed out or returned no video (status={state})",
                    "task_id": task_id,
                }

            resolved_dir = resolve_video_output_dir(output_dir, output_path, DEFAULT_OUTPUT_DIR, ())
            destination = (
                Path(output_path).expanduser()
                if output_path
                else Path(resolved_dir) / f"orcarouter_{task_id}.mp4"
            )
            size_bytes = _download(result_url, destination, timeout=timeout)
            elapsed = time.monotonic() - started
            return {
                "success": True,
                "provider": "orcarouter",
                "model": resolved_model,
                "task_id": task_id,
                "output_path": str(destination),
                "generation_type": generation_type,
                "aspect_ratio": aspect_ratio,
                "duration": int(payload["metadata"]["duration"]),
                "size_bytes": size_bytes,
                "elapsed_seconds": round(elapsed, 1),
            }
        except ValueError as exc:
            return {"success": False, "error": f"OrcaRouter validation error: {exc}"}
        except (requests.RequestException, OSError) as exc:
            return {"success": False, "error": f"OrcaRouter error: {exc.__class__.__name__}: {exc}"}


__all__ = [
    "OrcaRouterVideoGeneratorTool",
    "OrcaRouterVideoSchema",
    "build_orcarouter_payload",
]
