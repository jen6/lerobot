#!/usr/bin/env python

"""
Recording session manager for web-based episode recording control.

This module provides a RecordingSession class that allows direct control
of robot recording from the web UI, including:
- Starting/stopping recording sessions
- Saving/discarding individual episodes
- Real-time status monitoring
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
try:
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
except ImportError:
    from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.io_utils import load_info
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.robot import Robot
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR


@dataclass
class RecordingConfig:
    robot_type: str
    robot_port: str
    robot_id: str
    robot_cameras: dict[str, Any]
    teleop_type: str | None = None
    teleop_port: str | None = None
    teleop_id: str | None = None
    dataset_repo_id: str | None = "default/dataset"
    dataset_task: str | None = "default task"
    fps: int = 30
    use_videos: bool = True
    num_episodes: int = 50
    resume: bool = False


def _ensure_choice_registered(*, base_module: str, choice_name: str) -> None:
    module_aliases = {
        "robots": {
            "so101_follower": "so_follower",
            "so100_follower": "so_follower",
            "bi_so100_follower": "bi_so_follower",
        },
        "teleoperators": {
            "so101_leader": "so_leader",
            "so100_leader": "so_leader",
            "bi_so100_leader": "bi_so_leader",
        },
    }

    module_name = module_aliases.get(base_module, {}).get(choice_name, choice_name)
    importlib.import_module(f"lerobot.{base_module}.{module_name}.config_{module_name}")


def _ensure_camera_type_registered(camera_type: str) -> str:
    if camera_type == "realsense":
        camera_type = "intelrealsense"

    if camera_type == "opencv":
        importlib.import_module("lerobot.cameras.opencv.configuration_opencv")
    elif camera_type == "intelrealsense":
        importlib.import_module("lerobot.cameras.realsense.configuration_realsense")

    return camera_type


def _coerce_index_or_path(value: Any) -> int | Path:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return 0
    try:
        return int(text)
    except Exception:
        return Path(text)


def _encode_jpeg_base64(image: Any, *, is_rgb: bool) -> str | None:
    if image is None:
        return None

    try:
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image_np = np.asarray(image)
    except Exception:
        return None

    if image_np.ndim == 3 and image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3, 4):
        image_np = np.transpose(image_np, (1, 2, 0))

    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            max_val = float(np.nanmax(image_np)) if image_np.size else 0.0
            if max_val <= 1.0:
                image_np = (image_np * 255.0).clip(0, 255)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    if image_np.ndim == 3 and image_np.shape[-1] == 3 and is_rgb:
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    image_np = np.ascontiguousarray(image_np)
    ok, buf = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


class RecordingSession:
    """Manages a robot recording session with direct control over episodes."""

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.robot: Robot | None = None
        self.teleop: Teleoperator | None = None
        self.dataset: LeRobotDataset | None = None

        self.is_recording = False
        self.is_episode_active = False
        self.current_episode_frames = 0
        self.total_episodes_recorded = 0

        self.teleop_action_processor: RobotProcessorPipeline | None = None
        self.robot_action_processor: RobotProcessorPipeline | None = None
        self.robot_observation_processor: RobotProcessorPipeline | None = None

        self._teleoperation_task: asyncio.Task | None = None
        self._session_stop_requested = False
        self._latest_frame: dict[str, Any] = {}
        self._latest_motor_data: dict[str, Any] = {}
        self._motor_data_history: list[dict[str, Any]] = []
        self._motor_history_max_length = 100
        self._latest_motor_telemetry: dict[str, dict[str, float | bool | None]] = {}
        self._initial_dataset_episodes = 0

    @property
    def target_episode_count_reached(self) -> bool:
        return self.has_dataset and self.total_episodes_recorded >= self.config.num_episodes

    @property
    def has_dataset(self) -> bool:
        return self.dataset is not None and bool(self.config.dataset_repo_id)

    async def start(self) -> dict[str, Any]:
        """Initialize and connect robot, teleop, and dataset."""
        if self.is_recording:
            raise RuntimeError("Recording session already active")

        try:
            _ensure_choice_registered(base_module="robots", choice_name=self.config.robot_type)
            robot_cfg_cls = RobotConfig.get_choice_class(self.config.robot_type)

            camera_configs: dict[str, CameraConfig] = {}
            for cam_name, raw_cfg in (self.config.robot_cameras or {}).items():
                if not isinstance(raw_cfg, dict):
                    raise ValueError(f"Camera config for '{cam_name}' must be a JSON object")

                camera_type = _ensure_camera_type_registered(str(raw_cfg.get("type", "opencv")))
                cam_cfg_cls = CameraConfig.get_choice_class(camera_type)

                if camera_type == "opencv":
                    camera_configs[cam_name] = cam_cfg_cls(
                        index_or_path=_coerce_index_or_path(raw_cfg.get("index_or_path", 0)),
                        fps=int(raw_cfg.get("fps", self.config.fps)),
                        width=int(raw_cfg.get("width", 640)),
                        height=int(raw_cfg.get("height", 480)),
                        fourcc=raw_cfg.get("fourcc") or None,
                    )
                elif camera_type == "intelrealsense":
                    camera_configs[cam_name] = cam_cfg_cls(
                        serial_number_or_name=str(raw_cfg.get("serial_number_or_name", "")),
                        fps=int(raw_cfg.get("fps", self.config.fps)),
                        width=int(raw_cfg.get("width", 640)),
                        height=int(raw_cfg.get("height", 480)),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {camera_type}")

            robot_cfg = robot_cfg_cls(
                id=self.config.robot_id,
                port=self.config.robot_port,
                cameras=camera_configs,
            )

            self.robot = make_robot_from_config(robot_cfg)
            self.robot.connect()

            if self.config.teleop_type and self.config.teleop_port:
                _ensure_choice_registered(base_module="teleoperators", choice_name=self.config.teleop_type)
                teleop_cfg_cls = TeleoperatorConfig.get_choice_class(self.config.teleop_type)
                teleop_cfg = teleop_cfg_cls(
                    id=self.config.teleop_id,
                    port=self.config.teleop_port,
                )
                self.teleop = make_teleoperator_from_config(teleop_cfg)
                self.teleop.connect()

            (
                self.teleop_action_processor,
                self.robot_action_processor,
                self.robot_observation_processor,
            ) = make_default_processors()

            if self.config.dataset_repo_id:
                dataset_features = combine_feature_dicts(
                    aggregate_pipeline_dataset_features(
                        self.teleop_action_processor,
                        create_initial_features(action=self.robot.action_features),
                        use_videos=self.config.use_videos,
                    ),
                    aggregate_pipeline_dataset_features(
                        self.robot_observation_processor,
                        create_initial_features(observation=self.robot.observation_features),
                        use_videos=self.config.use_videos,
                    ),
                )
                dataset_root = HF_LEROBOT_HOME / self.config.dataset_repo_id
                dataset_exists = (dataset_root / "meta" / "info.json").exists()
                local_dataset_is_empty = False

                if dataset_exists:
                    try:
                        dataset_info = load_info(dataset_root)
                        local_dataset_is_empty = (
                            int(dataset_info.get("total_episodes", 0)) == 0
                            and int(dataset_info.get("total_frames", 0)) == 0
                        )
                    except Exception:
                        local_dataset_is_empty = False

                if self.config.resume:
                    if not dataset_exists:
                        raise RuntimeError(
                            f"Resume requested but dataset does not exist: {self.config.dataset_repo_id}"
                        )

                    if local_dataset_is_empty:
                        # Treat an empty local dataset as a fresh local target instead of attempting Hub resume.
                        shutil.rmtree(dataset_root)
                        dataset_exists = False

                    if dataset_exists:
                        self.dataset = LeRobotDataset(self.config.dataset_repo_id, root=dataset_root)

                        if self.dataset.fps != self.config.fps:
                            raise RuntimeError(
                                "Resume requested with mismatched FPS "
                                f"({self.config.fps} requested, {self.dataset.fps} existing)"
                            )

                        existing_robot_type = self.dataset.meta.robot_type
                        if existing_robot_type and existing_robot_type != self.config.robot_type:
                            raise RuntimeError(
                                "Resume requested with mismatched robot type "
                                f"({self.config.robot_type} requested, {existing_robot_type} existing)"
                            )

                    else:
                        self.dataset = LeRobotDataset.create(
                            repo_id=self.config.dataset_repo_id,
                            fps=self.config.fps,
                            features=dataset_features,
                            robot_type=self.config.robot_type,
                            use_videos=self.config.use_videos,
                        )
                else:
                    if dataset_exists and not local_dataset_is_empty:
                        raise RuntimeError(
                            "Dataset already exists. Enable resume to append new episodes to it."
                        )

                    if dataset_exists and local_dataset_is_empty:
                        shutil.rmtree(dataset_root)

                    self.dataset = LeRobotDataset.create(
                        repo_id=self.config.dataset_repo_id,
                        fps=self.config.fps,
                        features=dataset_features,
                        robot_type=self.config.robot_type,
                        use_videos=self.config.use_videos,
                    )

                self._initial_dataset_episodes = self.dataset.num_episodes
            else:
                self.dataset = None
                self._initial_dataset_episodes = 0

            self.is_recording = True
            self.is_episode_active = False
            self.current_episode_frames = 0
            self.total_episodes_recorded = 0
            self._session_stop_requested = False

            self._teleoperation_task = asyncio.create_task(self._teleoperation_loop())

            return {
                "status": "started",
                "dataset_repo_id": self.config.dataset_repo_id,
                "fps": self.config.fps,
                "has_dataset": self.has_dataset,
                "resume": self.config.resume,
                "dataset_total_episodes": self.dataset.num_episodes if self.dataset is not None else 0,
                "session_target_episodes": self.config.num_episodes if self.has_dataset else 0,
            }

        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start recording session: {e}") from e

    async def stop(self) -> dict[str, Any]:
        """Stop recording and cleanup resources."""
        if not self.is_recording:
            return {"status": "not_active"}

        self._session_stop_requested = True

        if self._teleoperation_task and not self._teleoperation_task.done():
            self._teleoperation_task.cancel()
            try:
                await self._teleoperation_task
            except asyncio.CancelledError:
                pass

        if self.is_episode_active and self.dataset is not None:
            self.dataset.clear_episode_buffer()
            self.is_episode_active = False

        await self.cleanup()

        result = {
            "status": "stopped",
            "total_episodes_recorded": self.total_episodes_recorded,
        }

        self.is_recording = False
        return result

    async def start_episode(self) -> dict[str, Any]:
        """Start recording a new episode."""
        if not self.is_recording:
            raise RuntimeError("Recording session not active")
        if self.dataset is None:
            raise RuntimeError("Episode control is unavailable without a dataset")

        if self.is_episode_active:
            raise RuntimeError("Episode already active")
        if self.target_episode_count_reached:
            raise RuntimeError(f"Target number of episodes already reached ({self.config.num_episodes})")

        self.is_episode_active = True
        self.current_episode_frames = 0

        return {
            "status": "episode_started",
            "episode_index": self.dataset.num_episodes,
        }

    async def stop_episode(self) -> dict[str, Any]:
        """Stop the current episode without saving."""
        if not self.is_episode_active:
            return {"status": "no_active_episode"}

        self.is_episode_active = False

        return {
            "status": "episode_stopped",
            "frames_recorded": self.current_episode_frames,
        }

    async def save_episode(self) -> dict[str, Any]:
        """Save the current episode to the dataset."""
        if not self.is_episode_active:
            raise RuntimeError("No active episode to save")
        if self.dataset is None:
            raise RuntimeError("Saving is unavailable without a dataset")
        if self.current_episode_frames <= 0:
            raise RuntimeError("Record at least one frame before saving the episode")

        await self.stop_episode()

        try:
            self.dataset.save_episode()
            self.total_episodes_recorded += 1

            return {
                "status": "episode_saved",
                "episode_index": self.dataset.num_episodes - 1,
                "frames": self.current_episode_frames,
                "total_episodes": self.total_episodes_recorded,
                "dataset_total_episodes": self.dataset.num_episodes,
                "session_target_episodes": self.config.num_episodes,
                "target_reached": self.target_episode_count_reached,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to save episode: {e}") from e

    async def discard_episode(self) -> dict[str, Any]:
        """Discard the current episode without saving."""
        if not self.is_episode_active:
            raise RuntimeError("No active episode to discard")
        if self.dataset is None:
            raise RuntimeError("Discarding is unavailable without a dataset")

        await self.stop_episode()

        self.dataset.clear_episode_buffer()

        return {
            "status": "episode_discarded",
            "frames_discarded": self.current_episode_frames,
        }

    async def get_status(self) -> dict[str, Any]:
        """Get current recording status."""
        return {
            "is_recording": self.is_recording,
            "is_episode_active": self.is_episode_active,
            "current_episode_frames": self.current_episode_frames,
            "total_episodes_recorded": self.total_episodes_recorded,
            "has_dataset": self.has_dataset,
            "dataset_total_episodes": self.dataset.num_episodes if self.dataset is not None else 0,
            "dataset_total_frames": self.dataset.meta.total_frames if self.dataset is not None else 0,
            "initial_dataset_episodes": self._initial_dataset_episodes,
            "session_target_episodes": self.config.num_episodes if self.has_dataset else 0,
            "remaining_episodes": max(self.config.num_episodes - self.total_episodes_recorded, 0)
            if self.has_dataset
            else 0,
            "target_reached": self.target_episode_count_reached,
            "resume": self.config.resume,
        }

    async def get_latest_frame(self) -> dict[str, str]:
        """Return the latest camera frames as base64-encoded JPEGs keyed by camera name."""
        if not self.robot or not hasattr(self.robot, "cameras"):
            return {}

        frames: dict[str, str] = {}
        cameras = getattr(self.robot, "cameras", {}) or {}

        for cam_key, cam in cameras.items():
            image = self._latest_frame.get(cam_key)
            if image is None:
                try:
                    image = cam.async_read()
                except Exception:
                    continue

            is_rgb = True
            try:
                cfg = getattr(getattr(self.robot, "config", None), "cameras", {}).get(cam_key)
                color_mode = getattr(cfg, "color_mode", None)
                if color_mode == ColorMode.BGR:
                    is_rgb = False
            except Exception:
                pass

            encoded = _encode_jpeg_base64(image, is_rgb=is_rgb)
            if encoded is not None:
                frames[cam_key] = encoded

        return frames

    async def get_latest_motor_data(self) -> dict[str, Any]:
        """Return the latest motor data including observation and action joint positions."""
        return self._latest_motor_data.copy()

    async def get_motor_data_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent motor data history for plotting."""
        return self._motor_data_history[-limit:]

    def _extract_motor_positions(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract motor position values from observation/action data."""
        motor_positions = {}
        for key, value in data.items():
            if key.endswith(".pos") and isinstance(value, (int, float)):
                motor_positions[key] = float(value)
        return motor_positions

    def _safe_sync_read(self, data_name: str) -> dict[str, float | bool] | None:
        bus = getattr(self.robot, "bus", None)
        if bus is None or not hasattr(bus, "sync_read"):
            return None

        try:
            values = bus.sync_read(data_name)
        except Exception:
            return None

        normalized: dict[str, float | bool] = {}
        for motor, value in values.items():
            if data_name == "Moving":
                normalized[motor] = bool(value)
            elif isinstance(value, (int, float)):
                normalized[motor] = float(value)
        return normalized

    def _collect_motor_telemetry(self) -> dict[str, dict[str, float | bool | None]]:
        telemetry_sources: dict[str, dict[str, float | bool] | None] = {
            "velocity": self._safe_sync_read("Present_Velocity"),
            "load": self._safe_sync_read("Present_Load"),
            "current": self._safe_sync_read("Present_Current"),
            "temperature": self._safe_sync_read("Present_Temperature"),
            "voltage": self._safe_sync_read("Present_Voltage"),
            "moving": self._safe_sync_read("Moving"),
        }

        if telemetry_sources["voltage"]:
            telemetry_sources["voltage"] = {
                motor: float(value) / 10.0 for motor, value in telemetry_sources["voltage"].items()
            }

        all_motors: set[str] = set()
        for values in telemetry_sources.values():
            if values:
                all_motors.update(values.keys())

        telemetry: dict[str, dict[str, float | bool | None]] = {}
        for motor in all_motors:
            telemetry[motor] = {
                metric: (values.get(motor) if values else None) for metric, values in telemetry_sources.items()
            }
        return telemetry

    def _do_teleoperation_step(self) -> dict[str, Any] | None:
        """
        Perform one teleoperation step synchronously.
        Returns processed data for recording, or None if teleop is not active.
        This runs in a thread pool to avoid blocking the event loop.
        """
        if not self.robot:
            return None

        obs = self.robot.get_observation()

        if hasattr(self.robot, "cameras"):
            for key, value in obs.items():
                if key in self.robot.cameras:
                    self._latest_frame[key] = value

        obs_motor_positions = self._extract_motor_positions(obs)
        motor_telemetry = self._collect_motor_telemetry()
        if motor_telemetry:
            self._latest_motor_telemetry = motor_telemetry
        else:
            motor_telemetry = self._latest_motor_telemetry

        if self.teleop:
            act = self.teleop.get_action()
            act_processed_teleop = self.teleop_action_processor((act, obs))
            robot_action_to_send = self.robot_action_processor((act_processed_teleop, obs))
            self.robot.send_action(robot_action_to_send)

            action_motor_positions = self._extract_motor_positions(act_processed_teleop)

            motor_data = {
                "timestamp": time.time(),
                "observation": obs_motor_positions,
                "action": action_motor_positions,
                "telemetry": motor_telemetry,
                "is_episode_active": self.is_episode_active,
                "frame_index": self.current_episode_frames,
            }
            self._latest_motor_data = motor_data

            self._motor_data_history.append(motor_data)
            if len(self._motor_data_history) > self._motor_history_max_length:
                self._motor_data_history = self._motor_data_history[-self._motor_history_max_length :]

            return {
                "obs": obs,
                "act_processed_teleop": act_processed_teleop,
            }

        motor_data = {
            "timestamp": time.time(),
            "observation": obs_motor_positions,
            "action": {},
            "telemetry": motor_telemetry,
            "is_episode_active": self.is_episode_active,
            "frame_index": self.current_episode_frames,
        }
        self._latest_motor_data = motor_data

        self._motor_data_history.append(motor_data)
        if len(self._motor_data_history) > self._motor_history_max_length:
            self._motor_data_history = self._motor_data_history[-self._motor_history_max_length :]

        return None

    async def _teleoperation_loop(self):
        """
        Continuous loop that runs while the session is active.
        Handles teleoperation (robot follows teleop device) and camera frame capture.
        Only records data to dataset when an episode is active.
        """
        if not self.robot:
            raise RuntimeError("Robot not initialized")

        dt = 1.0 / self.config.fps

        try:
            while not self._session_stop_requested:
                start_time = time.perf_counter()

                step_result = await asyncio.to_thread(self._do_teleoperation_step)

                if step_result and self.is_episode_active and self.dataset is not None:
                    obs = step_result["obs"]
                    act_processed_teleop = step_result["act_processed_teleop"]

                    obs_processed = self.robot_observation_processor(obs)
                    observation_frame = build_dataset_frame(
                        self.dataset.features, obs_processed, prefix=OBS_STR
                    )
                    action_frame = build_dataset_frame(
                        self.dataset.features, act_processed_teleop, prefix=ACTION
                    )
                    frame = {**observation_frame, **action_frame, "task": self.config.dataset_task or ""}
                    self.dataset.add_frame(frame)
                    self.current_episode_frames += 1

                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logging.info("Teleoperation loop cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in teleoperation loop: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.dataset is not None:
            try:
                self.dataset.finalize()
            except Exception as e:
                logging.warning(f"Error finalizing dataset: {e}")

        if self.robot:
            try:
                self.robot.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting robot: {e}")
            self.robot = None

        if self.teleop:
            try:
                self.teleop.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting teleop: {e}")
            self.teleop = None

        self.dataset = None
