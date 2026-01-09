#!/usr/bin/env python

from __future__ import annotations

import argparse
import asyncio
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

    def _get_lerobot_home() -> Path:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return HF_LEROBOT_HOME

    async def index(request):
        if not ui_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"UI file not found: {ui_path}. Set SO101_UI_PATH or run from repo root.",
            )
        return FileResponse(str(ui_path), headers={"Cache-Control": "no-store"})

    async def health(request):
        return JSONResponse({"ok": True})

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

    async def list_datasets(request):
        try:
            from lerobot.datasets.utils import load_info
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        lerobot_home = _get_lerobot_home()
        if not lerobot_home.exists():
            return JSONResponse({"datasets": []})

        datasets = []
        for dataset_dir in lerobot_home.iterdir():
            if not dataset_dir.is_dir():
                continue

            info_path = dataset_dir / "meta" / "info.json"
            if not info_path.exists():
                continue

            try:
                info = load_info(dataset_dir)
                datasets.append(
                    {
                        "repo_id": info.get("repo_id", dataset_dir.name),
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type", "unknown"),
                        "codebase_version": info.get("codebase_version", "unknown"),
                    }
                )
            except Exception:
                continue

        return JSONResponse({"datasets": datasets})

    async def get_dataset_info(request):
        try:
            from lerobot.datasets.utils import load_info
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
        try:
            from lerobot.datasets.utils import load_episodes
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to import lerobot modules: {e}") from e

        repo_id = request.path_params["repo_id"].replace("__", "/")
        lerobot_home = _get_lerobot_home()
        dataset_dir = lerobot_home / repo_id

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {repo_id}")

        try:
            episodes_dataset = load_episodes(dataset_dir)
            episodes = []
            for i in range(len(episodes_dataset)):
                ep = episodes_dataset[i]
                episode_info = {
                    "episode_index": int(ep["episode_index"]),
                    "length": int(ep["length"]),
                    "tasks": ep["tasks"] if "tasks" in ep else [],
                }
                episodes.append(episode_info)

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

        try:
            from lerobot.scripts.recording_session import RecordingConfig, RecordingSession

            body = await request.json()

            config = RecordingConfig(
                robot_type=body.get("robot_type", "so101_follower"),
                robot_port=body.get("robot_port", "/dev/ttyACM0"),
                robot_id=body.get("robot_id", "follower"),
                robot_cameras=body.get("robot_cameras", {}),
                teleop_type=body.get("teleop_type"),
                teleop_port=body.get("teleop_port"),
                teleop_id=body.get("teleop_id"),
                dataset_repo_id=body.get("dataset_repo_id", "default/dataset"),
                dataset_task=body.get("dataset_task", "default task"),
                fps=body.get("fps", 30),
                use_videos=body.get("use_videos", True),
            )

            recording_session = RecordingSession(config)
            result = await recording_session.start()

            return JSONResponse(result)
        except Exception as e:  # noqa: BLE001
            recording_session = None
            raise HTTPException(status_code=500, detail=f"Failed to start recording: {e}") from e

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

    async def recording_motor_data(request):
        if not recording_session:
            return JSONResponse({"motor_data": None, "message": "No active recording session"})

        try:
            motor_data = await recording_session.get_latest_motor_data()
            return JSONResponse({"motor_data": motor_data})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get motor data: {e}") from e

    async def recording_motor_history(request):
        if not recording_session:
            return JSONResponse({"history": [], "message": "No active recording session"})

        try:
            limit = int(request.query_params.get("limit", 100))
            history = await recording_session.get_motor_data_history(limit=limit)
            return JSONResponse({"history": history})
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to get motor history: {e}") from e

    async def stop_process(request):
        process_id = request.path_params["process_id"]
        mp = processes.get(process_id)
        if mp is None:
            raise HTTPException(status_code=404, detail="process not found")

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
        return JSONResponse({"stopped": True})

    async def ws_ping(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    async def ws_camera_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            if not recording_session or not recording_session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active recording session"})
                await websocket.close()
                return

            stream_fps = 15
            dt = 1.0 / stream_fps

            while recording_session and recording_session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    frames = await recording_session.get_latest_frame()

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

    async def ws_motor_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            if not recording_session or not recording_session.is_recording:
                await websocket.send_json({"type": "error", "message": "No active recording session"})
                await websocket.close()
                return

            stream_fps = 30
            dt = 1.0 / stream_fps

            while recording_session and recording_session.is_recording:
                start_time = asyncio.get_event_loop().time()

                try:
                    motor_data = await recording_session.get_latest_motor_data()

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

    async def ws_run(websocket: WebSocket):
        await websocket.accept()

        async def send(evt: dict[str, Any]):
            await websocket.send_json(evt)

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
        Route("/api/ports", endpoint=list_ports, methods=["GET"]),
        Route("/api/cameras", endpoint=list_cameras, methods=["GET"]),
        Route("/api/datasets", endpoint=list_datasets, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}/info", endpoint=get_dataset_info, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}/episodes", endpoint=list_episodes, methods=["GET"]),
        Route("/api/datasets/{repo_id:path}/episodes", endpoint=delete_episodes, methods=["DELETE"]),
        Route(
            "/api/datasets/{repo_id:path}/episodes/{episode_idx:int}/preview",
            endpoint=get_episode_preview,
            methods=["GET"],
        ),
        Route("/api/recording/start", endpoint=recording_start, methods=["POST"]),
        Route("/api/recording/stop", endpoint=recording_stop, methods=["POST"]),
        Route("/api/recording/start-episode", endpoint=recording_start_episode, methods=["POST"]),
        Route("/api/recording/stop-episode", endpoint=recording_stop_episode, methods=["POST"]),
        Route("/api/recording/save-episode", endpoint=recording_save_episode, methods=["POST"]),
        Route("/api/recording/discard-episode", endpoint=recording_discard_episode, methods=["POST"]),
        Route("/api/recording/status", endpoint=recording_status, methods=["GET"]),
        Route("/api/recording/motor-data", endpoint=recording_motor_data, methods=["GET"]),
        Route("/api/recording/motor-history", endpoint=recording_motor_history, methods=["GET"]),
        Route("/api/stop/{process_id}", endpoint=stop_process, methods=["POST"]),
        WebSocketRoute("/ws/ping", endpoint=ws_ping),
        WebSocketRoute("/ws/camera-stream", endpoint=ws_camera_stream),
        WebSocketRoute("/ws/motor-stream", endpoint=ws_motor_stream),
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
