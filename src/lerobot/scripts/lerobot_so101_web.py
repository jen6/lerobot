#!/usr/bin/env python

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _try_import_starlette():
    try:
        from starlette.applications import Starlette
        from starlette.exceptions import HTTPException
        from starlette.responses import FileResponse, JSONResponse
        from starlette.routing import Route, WebSocketRoute
        from starlette.staticfiles import StaticFiles
        from starlette.websockets import WebSocket, WebSocketDisconnect

        return (
            Starlette,
            HTTPException,
            FileResponse,
            JSONResponse,
            Route,
            WebSocketRoute,
            StaticFiles,
            WebSocket,
            WebSocketDisconnect,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing optional web dependencies. Install with: pip install -e '.[web]'") from e


def _repo_root() -> Path:
    search_roots: list[Path] = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    seen: set[Path] = set()

    for root in search_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)

        if (root / "so101_setup.html").exists() or (root / "pyproject.toml").exists():
            return root

    return Path(__file__).resolve().parents[3]


def _default_ui_path() -> Path:
    return _repo_root() / "so101_setup.html"


def _normalize_cmd_text(command: str) -> list[str]:
    import shlex

    cmd = command.replace("\\\n", " ").replace("\\\r\n", " ").strip()
    cmd = " ".join(cmd.split())
    return shlex.split(cmd)


ALLOWED_BINS: set[str] = {
    "lerobot-calibrate",
    "lerobot-dataset-viz",
    "lerobot-find-cameras",
    "lerobot-info",
    "lerobot-record",
    "lerobot-replay",
    "lerobot-setup-motors",
    "lerobot-teleoperate",
    "lerobot-train",
}


@dataclass
class ManagedProcess:
    id: str
    argv: list[str]
    proc: asyncio.subprocess.Process
    master_fd: int | None = None
    start_new_session: bool = False


def _pty_supported() -> bool:
    if os.name != "posix":
        return False
    try:
        import pty  # noqa: F401

        return True
    except Exception:
        return False


def _maybe_set_winsize(fd: int, *, cols: int, rows: int) -> None:
    if os.name != "posix":
        return
    try:
        import fcntl
        import struct
        import termios

        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except Exception:
        return


def _default_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("COLORTERM", "truecolor")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("FORCE_COLOR", "1")
    env.setdefault("RICH_FORCE_TERMINAL", "1")
    env.setdefault("RICH_COLOR_SYSTEM", "truecolor")
    return env


class CameraPreviewSession:
    def __init__(self):
        self.cameras: dict[str, Any] = {}
        self.camera_configs: dict[str, Any] = {}
        self.is_active = False
        self._capture_task: asyncio.Task | None = None
        self._latest_frames: dict[str, Any] = {}
        self._stop_requested = False
        self._fps = 15

    async def start(self, raw_camera_configs: dict[str, Any], *, fps: int = 15) -> dict[str, Any]:
        if self.is_active:
            await self.stop()

        if not raw_camera_configs:
            raise ValueError("robot_cameras is required")

        try:
            from lerobot.cameras.configs import CameraConfig
            from lerobot.cameras.utils import make_cameras_from_configs
            from lerobot.scripts.recording_session import _coerce_index_or_path, _ensure_camera_type_registered

            camera_configs: dict[str, Any] = {}
            for cam_name, raw_cfg in raw_camera_configs.items():
                if not isinstance(raw_cfg, dict):
                    raise ValueError(f"Camera config for '{cam_name}' must be a JSON object")

                camera_type = _ensure_camera_type_registered(str(raw_cfg.get("type", "opencv")))
                cam_cfg_cls = CameraConfig.get_choice_class(camera_type)

                if camera_type == "opencv":
                    camera_configs[cam_name] = cam_cfg_cls(
                        index_or_path=_coerce_index_or_path(raw_cfg.get("index_or_path", 0)),
                        fps=int(raw_cfg.get("fps", fps)),
                        width=int(raw_cfg.get("width", 640)),
                        height=int(raw_cfg.get("height", 480)),
                        fourcc=raw_cfg.get("fourcc") or None,
                    )
                elif camera_type == "intelrealsense":
                    camera_configs[cam_name] = cam_cfg_cls(
                        serial_number_or_name=str(raw_cfg.get("serial_number_or_name", "")),
                        fps=int(raw_cfg.get("fps", fps)),
                        width=int(raw_cfg.get("width", 640)),
                        height=int(raw_cfg.get("height", 480)),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {camera_type}")

            self.cameras = make_cameras_from_configs(camera_configs)
            for cam in self.cameras.values():
                cam.connect()

            self.camera_configs = camera_configs
            self._fps = max(1, min(int(fps), 60))
            self._stop_requested = False
            self._latest_frames = {}
            self.is_active = True
            self._capture_task = asyncio.create_task(self._capture_loop())

            return {
                "status": "started",
                "camera_names": sorted(self.cameras.keys()),
                "fps": self._fps,
            }
        except Exception:
            await self.cleanup()
            raise

    async def stop(self) -> dict[str, Any]:
        if not self.is_active and not self.cameras:
            return {"status": "not_active"}

        self._stop_requested = True
        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._capture_task

        await self.cleanup()
        return {"status": "stopped"}

    async def get_latest_frame(self) -> dict[str, str]:
        if not self.is_active:
            return {}

        from lerobot.cameras.configs import ColorMode
        from lerobot.scripts.recording_session import _encode_jpeg_base64

        frames: dict[str, str] = {}
        latest_frames = self._latest_frames
        if not latest_frames and self.cameras:
            latest_frames = await asyncio.to_thread(self._read_frames_once)

        for cam_key, image in latest_frames.items():
            cfg = self.camera_configs.get(cam_key)
            color_mode = getattr(cfg, "color_mode", None)
            encoded = _encode_jpeg_base64(image, is_rgb=color_mode != ColorMode.BGR)
            if encoded is not None:
                frames[cam_key] = encoded

        return frames

    def _read_frames_once(self) -> dict[str, Any]:
        frames: dict[str, Any] = {}
        for cam_key, cam in self.cameras.items():
            try:
                frames[cam_key] = cam.async_read()
            except Exception:
                continue
        return frames

    async def _capture_loop(self) -> None:
        dt = 1.0 / self._fps
        try:
            while not self._stop_requested:
                start_time = asyncio.get_running_loop().time()
                frames = await asyncio.to_thread(self._read_frames_once)
                if frames:
                    self._latest_frames = frames

                elapsed = asyncio.get_running_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            raise

    async def cleanup(self) -> None:
        for cam in self.cameras.values():
            with suppress(Exception):
                cam.disconnect()
        self.cameras = {}
        self.camera_configs = {}
        self._latest_frames = {}
        self._capture_task = None
        self.is_active = False
        self._stop_requested = False


def create_app(ui_path: Path, static_dir: Path | None = None):
    (
        Starlette,
        HTTPException,
        FileResponse,
        JSONResponse,
        Route,
        WebSocketRoute,
        StaticFiles,
        WebSocket,
        WebSocketDisconnect,
    ) = _try_import_starlette()

    processes: dict[str, ManagedProcess] = {}
    recording_session = None
    teleoperation_session = None
    eval_session = None
    camera_preview_session = CameraPreviewSession()

    def _get_lerobot_home() -> Path:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return HF_LEROBOT_HOME

    def _camera_settings_path() -> Path:
        return _get_lerobot_home() / "so101_web_camera_settings.json"

    def _default_camera_settings() -> dict[str, dict[str, Any]]:
        return {
            "front": {
                "type": "opencv",
                "index_or_path": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
            }
        }

    def _normalize_saved_camera_settings(raw_camera_configs: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw_camera_configs, dict):
            return _default_camera_settings()

        normalized: dict[str, dict[str, Any]] = {}
        for index, (raw_name, raw_cfg) in enumerate(raw_camera_configs.items()):
            if not isinstance(raw_cfg, dict):
                continue

            camera_name = str(raw_name).strip() or f"camera{index}"
            camera_type = str(raw_cfg.get("type", "opencv")).strip().lower()
            if camera_type == "realsense":
                camera_type = "intelrealsense"

            width = int(raw_cfg.get("width", 640))
            height = int(raw_cfg.get("height", 480))
            fps = int(raw_cfg.get("fps", 30))

            if camera_type == "intelrealsense":
                normalized[camera_name] = {
                    "type": "intelrealsense",
                    "serial_number_or_name": str(raw_cfg.get("serial_number_or_name", "")).strip(),
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            else:
                index_or_path = raw_cfg.get("index_or_path", 0)
                if isinstance(index_or_path, str):
                    stripped = index_or_path.strip()
                    if stripped.lstrip("-").isdigit():
                        index_or_path = int(stripped)
                    else:
                        index_or_path = stripped

                normalized[camera_name] = {
                    "type": "opencv",
                    "index_or_path": index_or_path,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "fourcc": raw_cfg.get("fourcc") or None,
                }

        return normalized or _default_camera_settings()

    def _load_camera_settings() -> dict[str, dict[str, Any]]:
        settings_path = _camera_settings_path()
        if not settings_path.exists():
            return _default_camera_settings()

        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return _default_camera_settings()

        return _normalize_saved_camera_settings(payload.get("robot_cameras"))

    def _save_camera_settings(raw_camera_configs: Any) -> dict[str, dict[str, Any]]:
        camera_settings = _normalize_saved_camera_settings(raw_camera_configs)
        settings_path = _camera_settings_path()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            json.dumps({"robot_cameras": camera_settings}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return camera_settings

    async def index(request):
        if not ui_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"UI file not found: {ui_path}. Set SO101_UI_PATH or run from repo root.",
            )
        return FileResponse(str(ui_path), headers={"Cache-Control": "no-store"})

    async def health(request):
        return JSONResponse({"ok": True})

    def _active_camera_motor_session():
        if recording_session and recording_session.is_recording:
            return recording_session
        if eval_session and eval_session.is_recording:
            return eval_session
        return None

    async def list_ports(request):
        try:
            from serial.tools import list_ports
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"pyserial not available: {e}") from e

        ports = []
        for p in list_ports.comports():
            ports.append(
                {
                    "device": p.device,
                    "description": getattr(p, "description", ""),
                    "hwid": getattr(p, "hwid", ""),
                    "vid": getattr(p, "vid", None),
                    "pid": getattr(p, "pid", None),
                    "serial_number": getattr(p, "serial_number", None),
                    "manufacturer": getattr(p, "manufacturer", None),
                    "product": getattr(p, "product", None),
                }
            )
        return JSONResponse({"ports": ports})

    async def list_cameras(request):
        try:
            from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"cameras": [], "warning": f"OpenCV camera backend not available: {e}"})

        try:
            cameras = OpenCVCamera.find_cameras()
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"cameras": [], "warning": f"Failed to enumerate cameras: {e}"})

        return JSONResponse({"cameras": cameras})

    async def get_camera_settings(request):
        return JSONResponse({"robot_cameras": _load_camera_settings()})

    async def save_camera_settings(request):
        try:
            body = await request.json()
            camera_settings = _save_camera_settings(body.get("robot_cameras"))
            return JSONResponse({"status": "saved", "robot_cameras": camera_settings})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to save camera settings: {e}") from e

    async def list_hf_models(request):
        try:
            from huggingface_hub import HfApi
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import huggingface_hub: {e}") from e

        try:
            api = HfApi()
            whoami = api.whoami()
            username = (
                whoami.get("name")
                or whoami.get("user")
                or whoami.get("fullname")
                or whoami.get("email")
                or whoami.get("orgs", [{}])[0].get("name")
            )
            if not username:
                raise RuntimeError("Unable to resolve the authenticated Hugging Face username")

            models = []
            for model_info in api.list_models(author=username, full=True):
                sibling_names = sorted(
                    {
                        getattr(sibling, "rfilename", "")
                        for sibling in (getattr(model_info, "siblings", None) or [])
                        if getattr(sibling, "rfilename", "")
                    }
                )
                tags = list(getattr(model_info, "tags", None) or [])
                has_config = "config.json" in sibling_names
                looks_like_lerobot = has_config and any(
                    tag.lower().startswith("lerobot") or "robot" in tag.lower() for tag in tags
                )
                models.append(
                    {
                        "repo_id": getattr(model_info, "modelId", ""),
                        "private": bool(getattr(model_info, "private", False)),
                        "downloads": int(getattr(model_info, "downloads", 0) or 0),
                        "last_modified": (
                            getattr(model_info, "lastModified", None).isoformat()
                            if hasattr(getattr(model_info, "lastModified", None), "isoformat")
                            else getattr(model_info, "lastModified", None)
                        ),
                        "tags": tags,
                        "has_config": has_config,
                        "looks_like_lerobot": looks_like_lerobot,
                    }
                )

            models.sort(key=lambda item: str(item.get("last_modified") or ""), reverse=True)
            return JSONResponse({"authenticated": True, "username": username, "models": models})
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"authenticated": False, "models": [], "warning": str(e)})

    async def list_datasets(request):
        try:
            from lerobot.datasets.io_utils import load_info
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        lerobot_home = _get_lerobot_home()
        if not lerobot_home.exists():
            return JSONResponse({"datasets": []})

        datasets = []
        seen_dataset_dirs: set[Path] = set()
        for info_path in sorted(lerobot_home.rglob("meta/info.json")):
            dataset_dir = info_path.parent.parent
            if dataset_dir in seen_dataset_dirs:
                continue
            seen_dataset_dirs.add(dataset_dir)

            try:
                info = load_info(dataset_dir)
                repo_id = info.get("repo_id")
                if not repo_id:
                    repo_id = dataset_dir.relative_to(lerobot_home).as_posix()

                datasets.append(
                    {
                        "repo_id": repo_id,
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type", "unknown"),
                        "codebase_version": info.get("codebase_version", "unknown"),
                    }
                )
            except Exception:
                continue

        datasets.sort(key=lambda item: str(item.get("repo_id", "")).lower())

        return JSONResponse({"datasets": datasets})

    async def get_dataset_info(request):
        try:
            from lerobot.datasets.io_utils import load_info
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        info_path = dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        try:
            info = load_info(dataset_dir)
            return JSONResponse(info)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to load dataset info: {e}") from e

    async def list_episodes(request):
        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        try:
            import pandas as pd

            episode_files = sorted((dataset_dir / "meta" / "episodes").glob("*/*.parquet"))
            if not episode_files:
                raise FileNotFoundError(f"Episode metadata not found in {dataset_dir / 'meta' / 'episodes'}")

            frames = [pd.read_parquet(path, columns=["episode_index", "length", "tasks"]) for path in episode_files]
            episodes_df = pd.concat(frames, ignore_index=True)
            episodes = []
            for _, row in episodes_df.sort_values("episode_index").iterrows():
                tasks = row["tasks"] if "tasks" in episodes_df.columns else []
                if not isinstance(tasks, list):
                    tasks = [tasks] if tasks not in (None, "") else []
                episodes.append(
                    {
                        "episode_index": int(row["episode_index"]),
                        "length": int(row["length"]),
                        "tasks": tasks,
                    }
                )

            return JSONResponse({"episodes": episodes})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to load episodes: {e}") from e

    async def delete_episodes(request):
        try:
            from lerobot.datasets.dataset_tools import delete_episodes as delete_eps_tool
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        try:
            body = await request.json()
            episode_indices = body.get("episode_indices", [])

            if not episode_indices:
                raise HTTPException(status_code=400, detail="episode_indices is required")

            dataset = LeRobotDataset(repo_id, root=lerobot_home / repo_id)

            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as temp_dir:
                new_dataset = delete_eps_tool(
                    dataset=dataset,
                    episode_indices=episode_indices,
                    output_dir=temp_dir,
                    repo_id=repo_id,
                )

                backup_dir = dataset_dir.parent / f"{dataset_dir.name}_backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                dataset_dir.rename(backup_dir)

                shutil.copytree(temp_dir, dataset_dir)
                shutil.rmtree(backup_dir)

            return JSONResponse(
                {
                    "success": True,
                    "deleted_episodes": episode_indices,
                    "remaining_episodes": new_dataset.meta.total_episodes,
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to delete episodes: {e}") from e

    async def delete_dataset(request):
        nonlocal recording_session, eval_session

        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        if (
            recording_session
            and recording_session.is_recording
            and recording_session.config.dataset_repo_id == repo_id
        ):
            raise HTTPException(status_code=400, detail="Cannot delete a dataset while it is being recorded")
        if eval_session and eval_session.is_recording and eval_session.config.dataset_repo_id == repo_id:
            raise HTTPException(status_code=400, detail="Cannot delete a dataset while it is being evaluated")

        try:
            import shutil

            shutil.rmtree(dataset_dir)
            return JSONResponse({"success": True, "deleted_repo_id": repo_id})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}") from e

    async def upload_dataset(request):
        nonlocal recording_session, eval_session

        try:
            from lerobot.datasets.io_utils import load_info
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        if (
            recording_session
            and recording_session.is_recording
            and recording_session.config.dataset_repo_id == repo_id
        ):
            raise HTTPException(status_code=400, detail="Stop recording before uploading this dataset")
        if eval_session and eval_session.is_recording and eval_session.config.dataset_repo_id == repo_id:
            raise HTTPException(status_code=400, detail="Stop evaluation before uploading this dataset")

        try:
            try:
                body = await request.json()
            except Exception:
                body = {}

            info = load_info(dataset_dir)
            dataset = LeRobotDataset(repo_id, root=dataset_dir)
            dataset.push_to_hub(
                private=bool(body.get("private", False)),
                tags=info.get("tags"),
            )
            return JSONResponse(
                {
                    "success": True,
                    "repo_id": repo_id,
                    "private": bool(body.get("private", False)),
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {e}") from e

    async def get_episode_preview(request):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            import base64
            import io
            from PIL import Image
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import required modules: {e}") from e

        repo_id = request.path_params["repo_id"].replace("__", "/")
        episode_idx = int(request.path_params["episode_idx"])
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        try:
            dataset = LeRobotDataset(repo_id, root=dataset_dir, episodes=[episode_idx])

            if len(dataset) == 0:
                raise HTTPException(status_code=404, detail=f"Episode {episode_idx} not found")

            first_frame = dataset[0]

            camera_keys = [k for k in first_frame.keys() if k.startswith("observation.images.")]

            if not camera_keys:
                return JSONResponse({"frames": []})

            frames_data = []
            for cam_key in camera_keys:
                img_tensor = first_frame[cam_key]

                if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = img_tensor.cpu().numpy()

                if img_np.dtype != "uint8":
                    img_np = (img_np * 255).clip(0, 255).astype("uint8")

                img = Image.fromarray(img_np)

                img.thumbnail((320, 240), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                frames_data.append(
                    {
                        "camera": cam_key,
                        "image_base64": f"data:image/jpeg;base64,{img_b64}",
                    }
                )

            return JSONResponse(
                {
                    "episode_index": episode_idx,
                    "total_frames": len(dataset),
                    "frames": frames_data,
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get episode preview: {e}") from e

    async def recording_start(request):
        nonlocal recording_session

        if recording_session and recording_session.is_recording:
            raise HTTPException(status_code=400, detail="Recording session already active")
        if teleoperation_session and teleoperation_session.is_recording:
            raise HTTPException(status_code=400, detail="Teleoperation session already active")
        if eval_session and eval_session.is_recording:
            raise HTTPException(status_code=400, detail="Evaluation session already active")

        try:
            from lerobot.scripts.recording_session import RecordingConfig, RecordingSession

            body = await request.json()
            robot_cameras = body.get("robot_cameras") or _load_camera_settings()
            robot_cameras = _save_camera_settings(robot_cameras)

            config = RecordingConfig(
                robot_type=body.get("robot_type", "so101_follower"),
                robot_port=body.get("robot_port", "/dev/ttyACM0"),
                robot_id=body.get("robot_id", "follower"),
                robot_cameras=robot_cameras,
                teleop_type=body.get("teleop_type"),
                teleop_port=body.get("teleop_port"),
                teleop_id=body.get("teleop_id"),
                dataset_repo_id=body.get("dataset_repo_id", "default/dataset"),
                dataset_task=body.get("dataset_task", "default task"),
                fps=body.get("fps", 30),
                use_videos=body.get("use_videos", True),
                num_episodes=body.get("num_episodes", 50),
                resume=body.get("resume", False),
                finalize_every_n_episodes=body.get("finalize_every_n_episodes", 5),
            )

            recording_session = RecordingSession(config)
            result = await recording_session.start()

            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            recording_session = None
            raise HTTPException(status_code=500, detail=f"Failed to start recording: {e}") from e

    async def teleoperation_start(request):
        nonlocal teleoperation_session

        if teleoperation_session and teleoperation_session.is_recording:
            raise HTTPException(status_code=400, detail="Teleoperation session already active")
        if recording_session and recording_session.is_recording:
            raise HTTPException(status_code=400, detail="Recording session already active")
        if eval_session and eval_session.is_recording:
            raise HTTPException(status_code=400, detail="Evaluation session already active")

        try:
            from lerobot.scripts.recording_session import RecordingConfig, RecordingSession

            body = await request.json()
            robot_cameras = body.get("robot_cameras") or _load_camera_settings()
            robot_cameras = _save_camera_settings(robot_cameras)

            if camera_preview_session.is_active:
                await camera_preview_session.stop()

            config = RecordingConfig(
                robot_type=body.get("robot_type", "so101_follower"),
                robot_port=body.get("robot_port", "/dev/ttyACM0"),
                robot_id=body.get("robot_id", "follower"),
                robot_cameras=robot_cameras,
                teleop_type=body.get("teleop_type"),
                teleop_port=body.get("teleop_port"),
                teleop_id=body.get("teleop_id"),
                dataset_repo_id=None,
                dataset_task=None,
                fps=body.get("fps", 30),
                use_videos=False,
            )

            teleoperation_session = RecordingSession(config)
            result = await teleoperation_session.start()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            teleoperation_session = None
            raise HTTPException(status_code=500, detail=f"Failed to start teleoperation: {e}") from e

    async def eval_start(request):
        nonlocal eval_session

        if eval_session and eval_session.is_recording:
            raise HTTPException(status_code=400, detail="Evaluation session already active")
        if recording_session and recording_session.is_recording:
            raise HTTPException(status_code=400, detail="Recording session already active")
        if teleoperation_session and teleoperation_session.is_recording:
            raise HTTPException(status_code=400, detail="Teleoperation session already active")

        try:
            from lerobot.scripts.recording_session import EvalConfig, EvalSession

            body = await request.json()
            robot_cameras = body.get("robot_cameras") or _load_camera_settings()
            robot_cameras = _save_camera_settings(robot_cameras)

            if camera_preview_session.is_active:
                await camera_preview_session.stop()

            config = EvalConfig(
                robot_type=body.get("robot_type", "so101_follower"),
                robot_port=body.get("robot_port", "/dev/ttyACM0"),
                robot_id=body.get("robot_id", "follower"),
                robot_cameras=robot_cameras,
                dataset_repo_id=body.get("dataset_repo_id", ""),
                dataset_task=body.get("dataset_task", ""),
                policy_path=body.get("policy_path", ""),
                policy_device=body.get("policy_device", "cpu"),
                fps=body.get("fps", 30),
                use_videos=body.get("use_videos", True),
                num_episodes=body.get("num_episodes", 50),
                resume=body.get("resume", False),
                finalize_every_n_episodes=body.get("finalize_every_n_episodes", 5),
            )

            eval_session = EvalSession(config)
            result = await eval_session.start()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            eval_session = None
            raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {e}") from e

    async def camera_preview_start(request):
        try:
            body = await request.json()
            robot_cameras = body.get("robot_cameras") or _load_camera_settings()
            robot_cameras = _save_camera_settings(robot_cameras)
            result = await camera_preview_session.start(
                robot_cameras,
                fps=int(body.get("fps", 15)),
            )
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to start camera preview: {e}") from e

    async def camera_preview_stop(request):
        try:
            result = await camera_preview_session.stop()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to stop camera preview: {e}") from e

    async def camera_preview_status(request):
        return JSONResponse(
            {
                "is_active": camera_preview_session.is_active,
                "camera_names": sorted(camera_preview_session.cameras.keys()),
            }
        )

    async def recording_stop(request):
        nonlocal recording_session

        if not recording_session:
            return JSONResponse({"status": "no_session"})

        try:
            result = await recording_session.stop()
            recording_session = None
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to stop recording: {e}") from e

    async def eval_stop(request):
        nonlocal eval_session

        if not eval_session:
            return JSONResponse({"status": "no_session"})

        try:
            result = await eval_session.stop()
            eval_session = None
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to stop evaluation: {e}") from e

    async def teleoperation_stop(request):
        nonlocal teleoperation_session

        if not teleoperation_session:
            return JSONResponse({"status": "no_session"})

        try:
            result = await teleoperation_session.stop()
            teleoperation_session = None
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to stop teleoperation: {e}") from e

    async def recording_start_episode(request):
        if not recording_session:
            raise HTTPException(status_code=400, detail="No active recording session")

        try:
            result = await recording_session.start_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to start episode: {e}") from e

    async def recording_stop_episode(request):
        if not recording_session:
            raise HTTPException(status_code=400, detail="No active recording session")

        try:
            result = await recording_session.stop_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to stop episode: {e}") from e

    async def recording_save_episode(request):
        if not recording_session:
            raise HTTPException(status_code=400, detail="No active recording session")

        try:
            result = await recording_session.save_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to save episode: {e}") from e

    async def recording_discard_episode(request):
        if not recording_session:
            raise HTTPException(status_code=400, detail="No active recording session")

        try:
            result = await recording_session.discard_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to discard episode: {e}") from e

    async def eval_start_episode(request):
        if not eval_session:
            raise HTTPException(status_code=400, detail="No active evaluation session")

        try:
            result = await eval_session.start_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to start eval episode: {e}") from e

    async def eval_save_episode(request):
        if not eval_session:
            raise HTTPException(status_code=400, detail="No active evaluation session")

        try:
            result = await eval_session.save_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to save eval episode: {e}") from e

    async def eval_discard_episode(request):
        if not eval_session:
            raise HTTPException(status_code=400, detail="No active evaluation session")

        try:
            result = await eval_session.discard_episode()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to discard eval episode: {e}") from e

    async def recording_status(request):
        if not recording_session:
            return JSONResponse(
                {
                    "is_recording": False,
                    "is_episode_active": False,
                    "current_episode_frames": 0,
                    "total_episodes_recorded": 0,
                }
            )

        try:
            result = await recording_session.get_status()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get status: {e}") from e

    async def eval_status(request):
        if not eval_session:
            return JSONResponse(
                {
                    "is_recording": False,
                    "is_episode_active": False,
                    "current_episode_frames": 0,
                    "total_episodes_recorded": 0,
                    "has_dataset": False,
                    "has_policy": False,
                }
            )

        try:
            result = await eval_session.get_status()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get evaluation status: {e}") from e

    async def teleoperation_status(request):
        if not teleoperation_session:
            return JSONResponse(
                {
                    "is_recording": False,
                    "current_episode_frames": 0,
                    "total_episodes_recorded": 0,
                    "has_dataset": False,
                }
            )

        try:
            result = await teleoperation_session.get_status()
            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get teleoperation status: {e}") from e

    async def recording_motor_data(request):
        session = _active_camera_motor_session()
        if not session:
            return JSONResponse({"motor_data": None, "message": "No active recording session"})

        try:
            motor_data = await session.get_latest_motor_data()
            return JSONResponse({"motor_data": motor_data})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get motor data: {e}") from e

    async def teleoperation_motor_data(request):
        if not teleoperation_session:
            return JSONResponse({"motor_data": None, "message": "No active teleoperation session"})

        try:
            motor_data = await teleoperation_session.get_latest_motor_data()
            return JSONResponse({"motor_data": motor_data})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get teleoperation motor data: {e}") from e

    async def recording_motor_history(request):
        session = _active_camera_motor_session()
        if not session:
            return JSONResponse({"history": [], "message": "No active recording session"})

        try:
            limit = int(request.query_params.get("limit", 100))
            history = await session.get_motor_data_history(limit=limit)
            return JSONResponse({"history": history})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get motor history: {e}") from e

    async def teleoperation_motor_history(request):
        if not teleoperation_session:
            return JSONResponse({"history": [], "message": "No active teleoperation session"})

        try:
            limit = int(request.query_params.get("limit", 100))
            history = await teleoperation_session.get_motor_data_history(limit=limit)
            return JSONResponse({"history": history})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get teleoperation motor history: {e}") from e

    async def _stop_managed_process(process_id: str, mp: ManagedProcess) -> dict[str, Any]:
        try:
            if mp.start_new_session and mp.proc.pid:
                os.killpg(mp.proc.pid, signal.SIGINT)
            else:
                mp.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                mp.proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

        try:
            await asyncio.wait_for(mp.proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                if mp.start_new_session and mp.proc.pid:
                    os.killpg(mp.proc.pid, signal.SIGKILL)
                else:
                    mp.proc.kill()
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    mp.proc.kill()
                except ProcessLookupError:
                    pass
            await mp.proc.wait()

        processes.pop(process_id, None)
        return {"process_id": process_id, "stopped": True}

    async def _stop_all_runtime() -> dict[str, Any]:
        stopped_processes: list[dict[str, Any]] = []
        for process_id, mp in list(processes.items()):
            stopped_processes.append(await _stop_managed_process(process_id, mp))

        stopped_sessions: list[str] = []
        nonlocal_recording_session = recording_session
        nonlocal_teleoperation_session = teleoperation_session
        nonlocal_eval_session = eval_session

        if nonlocal_recording_session and nonlocal_recording_session.is_recording:
            await nonlocal_recording_session.stop()
            stopped_sessions.append("recording")
        if nonlocal_teleoperation_session and nonlocal_teleoperation_session.is_recording:
            await nonlocal_teleoperation_session.stop()
            stopped_sessions.append("teleoperation")
        if nonlocal_eval_session and nonlocal_eval_session.is_recording:
            await nonlocal_eval_session.stop()
            stopped_sessions.append("eval")
        if camera_preview_session.is_active:
            await camera_preview_session.stop()
            stopped_sessions.append("camera_preview")

        return {
            "stopped": True,
            "stopped_processes": stopped_processes,
            "stopped_sessions": stopped_sessions,
        }

    async def stop_process(request):
        nonlocal recording_session, teleoperation_session, eval_session
        process_id = request.path_params["process_id"]
        mp = processes.get(process_id)
        if mp is not None:
            result = await _stop_managed_process(process_id, mp)
            return JSONResponse(result)

        result = await _stop_all_runtime()
        recording_session = None if recording_session and not recording_session.is_recording else recording_session
        teleoperation_session = (
            None if teleoperation_session and not teleoperation_session.is_recording else teleoperation_session
        )
        eval_session = None if eval_session and not eval_session.is_recording else eval_session
        result["fallback"] = "stopped_all"
        result["requested_process_id"] = process_id
        return JSONResponse(result)

    async def stop_all(request):
        nonlocal recording_session, teleoperation_session, eval_session
        result = await _stop_all_runtime()
        recording_session = None
        teleoperation_session = None
        eval_session = None
        return JSONResponse(result)

    async def ws_ping(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    async def ws_camera_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            session = _active_camera_motor_session()
            if not session or not session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active recording or evaluation session"})
                await websocket.close()
                return

            stream_fps = 15
            dt = 1.0 / stream_fps

            while session and session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    frames = await session.get_latest_frame()

                    if frames:
                        await websocket.send_json({"type": "frames", "data": frames, "timestamp": start_time})
                    else:
                        await websocket.send_json(
                            {"type": "no_frames", "message": "No camera frames available yet"}
                        )

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to get frames: {e}"})
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": f"Stream error: {e}"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def ws_teleoperation_camera_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            if not teleoperation_session or not teleoperation_session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active teleoperation session"})
                await websocket.close()
                return

            stream_fps = 15
            dt = 1.0 / stream_fps

            while teleoperation_session and teleoperation_session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    frames = await teleoperation_session.get_latest_frame()

                    if frames:
                        await websocket.send_json({"type": "frames", "data": frames, "timestamp": start_time})
                    else:
                        await websocket.send_json(
                            {"type": "no_frames", "message": "No camera frames available yet"}
                        )

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to get frames: {e}"})
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": f"Stream error: {e}"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def ws_camera_preview(websocket: WebSocket):
        await websocket.accept()

        try:
            if not camera_preview_session.is_active:
                await websocket.send_json({"type": "error", "message": "Camera preview session is not active"})
                await websocket.close()
                return

            stream_fps = 15
            dt = 1.0 / stream_fps

            while camera_preview_session.is_active:
                start_time = asyncio.get_running_loop().time()

                try:
                    frames = await camera_preview_session.get_latest_frame()
                    if frames:
                        await websocket.send_json({"type": "frames", "data": frames, "timestamp": start_time})
                    else:
                        await websocket.send_json(
                            {"type": "no_frames", "message": "No camera frames available yet"}
                        )
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to get preview frames: {e}"})
                    break

                elapsed = asyncio.get_running_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": f"Preview stream error: {e}"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def ws_motor_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            session = _active_camera_motor_session()
            if not session or not session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active recording or evaluation session"})
                await websocket.close()
                return

            stream_fps = 30
            dt = 1.0 / stream_fps

            while session and session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    motor_data = await session.get_latest_motor_data()

                    if motor_data:
                        await websocket.send_json({"type": "motor_data", "data": motor_data})
                    else:
                        await websocket.send_json(
                            {"type": "no_data", "message": "No motor data available yet"}
                        )

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to get motor data: {e}"})
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": f"Stream error: {e}"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def ws_teleoperation_motor_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            if not teleoperation_session or not teleoperation_session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active teleoperation session"})
                await websocket.close()
                return

            stream_fps = 30
            dt = 1.0 / stream_fps

            while teleoperation_session and teleoperation_session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    motor_data = await teleoperation_session.get_latest_motor_data()

                    if motor_data:
                        await websocket.send_json({"type": "motor_data", "data": motor_data})
                    else:
                        await websocket.send_json({"type": "no_data", "message": "No motor data available yet"})

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to get motor data: {e}"})
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": f"Stream error: {e}"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def ws_run(websocket: WebSocket):
        await websocket.accept()

        async def send(evt: dict[str, Any]):
            try:
                await websocket.send_json(evt)
            except (WebSocketDisconnect, RuntimeError):
                return

        try:
            while True:
                msg = await websocket.receive_json()
                raw_command = str(msg.get("command", "")).strip()
                process_id = str(msg.get("id") or uuid.uuid4().hex)
                try:
                    cols = int(msg.get("cols") or 120)
                except Exception:
                    cols = 120
                try:
                    rows = int(msg.get("rows") or 30)
                except Exception:
                    rows = 30
                cols = max(20, min(cols, 400))
                rows = max(5, min(rows, 200))

                if not raw_command:
                    await send({"type": "error", "id": process_id, "message": "empty command"})
                    continue

                argv = _normalize_cmd_text(raw_command)
                if not argv:
                    await send({"type": "error", "id": process_id, "message": "empty argv"})
                    continue

                if argv[0] not in ALLOWED_BINS:
                    await send(
                        {
                            "type": "error",
                            "id": process_id,
                            "message": f"command not allowed: {argv[0]}",
                            "allowed": sorted(ALLOWED_BINS),
                        }
                    )
                    continue

                if process_id in processes:
                    await send({"type": "error", "id": process_id, "message": "process id already running"})
                    continue

                use_pty = _pty_supported()
                master_fd: int | None = None
                await send(
                    {
                        "type": "started",
                        "id": process_id,
                        "argv": argv,
                        "pty": use_pty,
                        "cols": cols,
                        "rows": rows,
                    }
                )

                if use_pty:
                    import pty

                    master_fd, slave_fd = pty.openpty()
                    _maybe_set_winsize(slave_fd, cols=cols, rows=rows)
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        cwd=str(_repo_root()),
                        stdin=slave_fd,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        env=_default_subprocess_env(),
                        start_new_session=True,
                    )
                    try:
                        os.close(slave_fd)
                    except OSError:
                        pass
                    try:
                        os.set_blocking(master_fd, False)
                    except Exception:
                        pass
                    processes[process_id] = ManagedProcess(
                        id=process_id,
                        argv=argv,
                        proc=proc,
                        master_fd=master_fd,
                        start_new_session=True,
                    )
                else:
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        cwd=str(_repo_root()),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env=_default_subprocess_env(),
                        start_new_session=True,
                    )
                    processes[process_id] = ManagedProcess(
                        id=process_id, argv=argv, proc=proc, master_fd=None, start_new_session=True
                    )

                try:
                    if master_fd is not None:
                        loop = asyncio.get_running_loop()
                        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
                        wait_task: asyncio.Task[None] | None = None

                        def on_readable() -> None:
                            assert master_fd is not None
                            try:
                                while True:
                                    chunk = os.read(master_fd, 4096)
                                    if not chunk:
                                        queue.put_nowait(None)
                                        return
                                    queue.put_nowait(chunk)
                            except BlockingIOError:
                                return
                            except OSError:
                                queue.put_nowait(None)
                                return

                        loop.add_reader(master_fd, on_readable)
                        try:

                            async def wait_and_signal_done() -> None:
                                await proc.wait()
                                queue.put_nowait(None)

                            wait_task = asyncio.create_task(wait_and_signal_done())
                            while True:
                                chunk = await queue.get()
                                if chunk is None:
                                    with suppress(Exception):
                                        while True:
                                            extra = os.read(master_fd, 4096)
                                            if not extra:
                                                break
                                            queue.put_nowait(extra)
                                    break
                                text = chunk.decode(errors="replace")
                                await send({"type": "output", "id": process_id, "data": text, "line": text})
                        finally:
                            if wait_task is not None:
                                wait_task.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await wait_task
                            loop.remove_reader(master_fd)
                            try:
                                os.close(master_fd)
                            except OSError:
                                pass
                    else:
                        assert proc.stdout is not None
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            text = line.decode(errors="replace")
                            await send({"type": "output", "id": process_id, "data": text, "line": text})
                finally:
                    rc = await proc.wait()
                    processes.pop(process_id, None)
                    await send({"type": "exit", "id": process_id, "return_code": rc})
        except WebSocketDisconnect:
            return

    routes = [
        Route("/", endpoint=index, methods=["GET"]),
        Route("/api/health", endpoint=health, methods=["GET"]),
        Route("/api/hf/models", endpoint=list_hf_models, methods=["GET"]),
        Route("/api/ports", endpoint=list_ports, methods=["GET"]),
        Route("/api/cameras", endpoint=list_cameras, methods=["GET"]),
        Route("/api/settings/cameras", endpoint=get_camera_settings, methods=["GET"]),
        Route("/api/settings/cameras", endpoint=save_camera_settings, methods=["POST"]),
        Route("/api/datasets", endpoint=list_datasets, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}", endpoint=delete_dataset, methods=["DELETE"]),
        Route("/api/datasets/{repo_id:path}/upload", endpoint=upload_dataset, methods=["POST"]),
        Route("/api/datasets/{repo_id:path}/info", endpoint=get_dataset_info, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}/episodes", endpoint=list_episodes, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}/episodes", endpoint=delete_episodes, methods=["DELETE"]),
        Route(
            "/api/datasets/{repo_id:path}/episodes/{episode_idx:int}/preview",
            endpoint=get_episode_preview,
            methods=["GET"],
        ),
        Route("/api/camera-preview/start", endpoint=camera_preview_start, methods=["POST"]),
        Route("/api/camera-preview/stop", endpoint=camera_preview_stop, methods=["POST"]),
        Route("/api/camera-preview/status", endpoint=camera_preview_status, methods=["GET"]),
        Route("/api/teleoperation/start", endpoint=teleoperation_start, methods=["POST"]),
        Route("/api/teleoperation/stop", endpoint=teleoperation_stop, methods=["POST"]),
        Route("/api/teleoperation/status", endpoint=teleoperation_status, methods=["GET"]),
        Route("/api/teleoperation/motor-data", endpoint=teleoperation_motor_data, methods=["GET"]),
        Route("/api/teleoperation/motor-history", endpoint=teleoperation_motor_history, methods=["GET"]),
        Route("/api/eval/start", endpoint=eval_start, methods=["POST"]),
        Route("/api/eval/stop", endpoint=eval_stop, methods=["POST"]),
        Route("/api/eval/start-episode", endpoint=eval_start_episode, methods=["POST"]),
        Route("/api/eval/save-episode", endpoint=eval_save_episode, methods=["POST"]),
        Route("/api/eval/discard-episode", endpoint=eval_discard_episode, methods=["POST"]),
        Route("/api/eval/status", endpoint=eval_status, methods=["GET"]),
        Route("/api/recording/start", endpoint=recording_start, methods=["POST"]),
        Route("/api/recording/stop", endpoint=recording_stop, methods=["POST"]),
        Route("/api/recording/start-episode", endpoint=recording_start_episode, methods=["POST"]),
        Route("/api/recording/stop-episode", endpoint=recording_stop_episode, methods=["POST"]),
        Route("/api/recording/save-episode", endpoint=recording_save_episode, methods=["POST"]),
        Route("/api/recording/discard-episode", endpoint=recording_discard_episode, methods=["POST"]),
        Route("/api/recording/status", endpoint=recording_status, methods=["GET"]),
        Route("/api/recording/motor-data", endpoint=recording_motor_data, methods=["GET"]),
        Route("/api/recording/motor-history", endpoint=recording_motor_history, methods=["GET"]),
        Route("/api/stop-all", endpoint=stop_all, methods=["POST"]),
        Route("/api/stop/{process_id}", endpoint=stop_process, methods=["POST"]),
        WebSocketRoute("/ws/ping", endpoint=ws_ping),
        WebSocketRoute("/ws/camera-stream", endpoint=ws_camera_stream),
        WebSocketRoute("/ws/teleoperation-camera-stream", endpoint=ws_teleoperation_camera_stream),
        WebSocketRoute("/ws/camera-preview", endpoint=ws_camera_preview),
        WebSocketRoute("/ws/motor-stream", endpoint=ws_motor_stream),
        WebSocketRoute("/ws/teleoperation-motor-stream", endpoint=ws_teleoperation_motor_stream),
        WebSocketRoute("/ws/run", endpoint=ws_run),
    ]

    app = Starlette(routes=routes)
    if static_dir and static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="SO-ARM101 local web UI (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SO101_WEB_PORT", "8000")))
    parser.add_argument("--ui", type=str, default=os.environ.get("SO101_UI_PATH", ""))
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    args = parser.parse_args()

    ui_path = Path(args.ui).expanduser() if args.ui else _default_ui_path()
    app = create_app(ui_path=ui_path, static_dir=None)

    try:
        import uvicorn
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing uvicorn. Install with: pip install -e '.[web]'") from e

    if args.open:
        import webbrowser

        webbrowser.open(f"http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", ws="wsproto")


if __name__ == "__main__":
    main()
