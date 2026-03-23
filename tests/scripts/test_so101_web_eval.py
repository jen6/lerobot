from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub

stub = types.ModuleType("lerobot.utils.io_utils")
stub.deserialize_json_into_object = lambda fpath, obj: obj  # noqa: E731
sys.modules.setdefault("lerobot.utils.io_utils", stub)

recording_session_stub = types.ModuleType("lerobot.scripts.recording_session")
sys.modules["lerobot.scripts.recording_session"] = recording_session_stub

from lerobot.scripts import lerobot_so101_web as web_module


@dataclass
class _FakeHfSibling:
    rfilename: str


@dataclass
class _FakeHfModel:
    modelId: str
    private: bool
    downloads: int
    lastModified: str
    tags: list[str]
    siblings: list[_FakeHfSibling]


class _FakeHfApi:
    def whoami(self):
        return {"name": "test-user"}

    def list_models(self, author: str, full: bool):
        assert author == "test-user"
        assert full is True
        return [
            _FakeHfModel(
                modelId="test-user/older-model",
                private=False,
                downloads=7,
                lastModified=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                tags=["vision"],
                siblings=[_FakeHfSibling("config.json")],
            ),
            _FakeHfModel(
                modelId="test-user/newer-model",
                private=True,
                downloads=42,
                lastModified=datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc),
                tags=["lerobot", "policy"],
                siblings=[_FakeHfSibling("config.json"), _FakeHfSibling("model.safetensors")],
            ),
        ]


class _FakeEvalSession:
    def __init__(self, config):
        self.config = config
        self.is_recording = False
        self.is_episode_active = False
        self.current_episode_frames = 0
        self.total_episodes_recorded = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.start_episode_calls = 0
        self.save_episode_calls = 0
        self.discard_episode_calls = 0

    async def start(self):
        self.start_calls += 1
        self.is_recording = True
        return {
            "status": "started",
            "dataset_repo_id": self.config.dataset_repo_id,
            "policy_path": self.config.policy_path,
            "policy_device": self.config.policy_device,
            "fps": self.config.fps,
            "has_dataset": True,
            "resume": self.config.resume,
            "dataset_total_episodes": 3,
            "session_target_episodes": self.config.num_episodes,
        }

    async def stop(self):
        self.stop_calls += 1
        self.is_recording = False
        self.is_episode_active = False
        return {"status": "stopped", "total_episodes_recorded": self.total_episodes_recorded}

    async def start_episode(self):
        self.start_episode_calls += 1
        self.is_episode_active = True
        self.current_episode_frames = 0
        return {"status": "episode_started", "episode_index": 3}

    async def save_episode(self):
        self.save_episode_calls += 1
        self.is_episode_active = False
        self.total_episodes_recorded += 1
        return {
            "status": "episode_saved",
            "episode_index": 3,
            "frames": 6,
            "total_episodes": self.total_episodes_recorded,
            "dataset_total_episodes": 4,
            "initial_dataset_episodes": 3,
            "session_target_episodes": self.config.num_episodes,
            "target_reached": False,
        }

    async def discard_episode(self):
        self.discard_episode_calls += 1
        self.is_episode_active = False
        return {"status": "episode_discarded", "frames_discarded": 4}

    async def get_status(self):
        return {
            "is_recording": self.is_recording,
            "is_episode_active": self.is_episode_active,
            "current_episode_frames": self.current_episode_frames,
            "total_episodes_recorded": self.total_episodes_recorded,
            "has_dataset": True,
            "dataset_total_episodes": 4,
            "dataset_total_frames": 120,
            "initial_dataset_episodes": 3,
            "session_target_episodes": self.config.num_episodes,
            "remaining_episodes": max(self.config.num_episodes - self.total_episodes_recorded, 0),
            "target_reached": False,
            "resume": self.config.resume,
            "has_policy": True,
            "policy_path": self.config.policy_path,
            "policy_device": self.config.policy_device,
            "mode": "eval",
        }


class _FakeEvalConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


recording_session_stub.EvalConfig = _FakeEvalConfig
recording_session_stub.EvalSession = _FakeEvalSession


class _FakeRequest:
    def __init__(self, *, body=None, path_params=None, query_params=None):
        self._body = body or {}
        self.path_params = path_params or {}
        self.query_params = query_params or {}

    async def json(self):
        return self._body


def _make_app(tmp_path: Path):
    ui_path = tmp_path / "so101_setup.html"
    ui_path.write_text("<html></html>", encoding="utf-8")
    return web_module.create_app(ui_path=ui_path, static_dir=None)


def _get_route(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def _response_json(response):
    return json.loads(response.body.decode("utf-8"))


def test_hf_models_route_lists_user_models(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    app = _make_app(tmp_path)
    list_hf_models = _get_route(app, "/api/hf/models", "GET")

    response = asyncio.run(list_hf_models(_FakeRequest()))
    payload = _response_json(response)

    assert payload["authenticated"] is True
    assert payload["username"] == "test-user"
    assert [model["repo_id"] for model in payload["models"]] == [
        "test-user/newer-model",
        "test-user/older-model",
    ]
    assert payload["models"][0]["looks_like_lerobot"] is True
    assert payload["models"][0]["has_config"] is True
    assert payload["models"][1]["private"] is False
    assert payload["models"][0]["last_modified"] == "2025-03-01T12:00:00+00:00"


def test_eval_routes_cover_lifecycle(monkeypatch, tmp_path):
    app = _make_app(tmp_path)
    start_eval = _get_route(app, "/api/eval/start", "POST")
    stop_eval = _get_route(app, "/api/eval/stop", "POST")
    start_episode = _get_route(app, "/api/eval/start-episode", "POST")
    save_episode = _get_route(app, "/api/eval/save-episode", "POST")
    discard_episode = _get_route(app, "/api/eval/discard-episode", "POST")
    eval_status = _get_route(app, "/api/eval/status", "GET")

    start = asyncio.run(
        start_eval(
            _FakeRequest(
                body={
                    "robot_type": "so101_follower",
                    "robot_port": "/dev/ttyACM0",
                    "robot_id": "follower",
                    "robot_cameras": {"front": {"type": "opencv", "index_or_path": 0}},
                    "dataset_repo_id": "test-user/eval_so101",
                    "dataset_task": "pick the cube",
                    "policy_path": "test-user/model",
                    "policy_device": "cpu",
                    "fps": 30,
                    "use_videos": True,
                    "num_episodes": 5,
                    "resume": False,
                }
            )
        )
    )
    status = asyncio.run(eval_status(_FakeRequest()))
    start_episode_response = asyncio.run(start_episode(_FakeRequest()))
    save_episode_response = asyncio.run(save_episode(_FakeRequest()))
    asyncio.run(start_episode(_FakeRequest()))
    discard_episode_response = asyncio.run(discard_episode(_FakeRequest()))
    stop = asyncio.run(stop_eval(_FakeRequest()))
    stopped_status = asyncio.run(eval_status(_FakeRequest()))

    assert _response_json(start)["status"] == "started"
    assert _response_json(start)["policy_path"] == "test-user/model"
    assert _response_json(status)["is_recording"] is True
    assert _response_json(status)["has_policy"] is True
    assert _response_json(start_episode_response) == {"status": "episode_started", "episode_index": 3}
    save_payload = _response_json(save_episode_response)
    assert save_payload["status"] == "episode_saved"
    assert save_payload["total_episodes"] == 1
    assert _response_json(discard_episode_response) == {"status": "episode_discarded", "frames_discarded": 4}
    assert _response_json(stop)["status"] == "stopped"
    assert _response_json(stopped_status)["is_recording"] is False
    assert _response_json(stopped_status)["has_policy"] is False
