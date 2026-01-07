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
import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor.robot_processor import RobotProcessorPipeline
from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR


@dataclass
class RecordingConfig:
    robot_type: str
    robot_port: str
    robot_id: str
    robot_cameras: dict[str, Any]
    teleop_type: str | None = None
    teleop_port: str | None = None
    teleop_id: str | None = None
    dataset_repo_id: str = "default/dataset"
    dataset_task: str = "default task"
    fps: int = 30
    use_videos: bool = True


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

        self._recording_task: asyncio.Task | None = None
        self._stop_requested = False
        self._latest_frame: dict[str, Any] = {}

    async def start(self) -> dict[str, Any]:
        """Initialize and connect robot, teleop, and dataset."""
        if self.is_recording:
            raise RuntimeError("Recording session already active")

        try:
            from lerobot.configs.robot import make_robot_from_config
            from lerobot.configs.teleoperator import make_teleoperator_from_config
            from lerobot.processor.robot_processor import make_default_processors
            from lerobot.processor.utils import (
                aggregate_pipeline_dataset_features,
                combine_feature_dicts,
                create_initial_features,
            )

            robot_config_dict = {
                "type": self.config.robot_type,
                "port": self.config.robot_port,
                "id": self.config.robot_id,
                "cameras": self.config.robot_cameras,
            }

            self.robot = make_robot_from_config(robot_config_dict)
            self.robot.connect()

            if self.config.teleop_type and self.config.teleop_port:
                teleop_config_dict = {
                    "type": self.config.teleop_type,
                    "port": self.config.teleop_port,
                    "id": self.config.teleop_id,
                }
                self.teleop = make_teleoperator_from_config(teleop_config_dict)
                self.teleop.connect()

            (
                self.teleop_action_processor,
                self.robot_action_processor,
                self.robot_observation_processor,
            ) = make_default_processors()

            dataset_features = combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=self.teleop_action_processor,
                    initial_features=create_initial_features(action=self.robot.action_features),
                    use_videos=self.config.use_videos,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=self.robot_observation_processor,
                    initial_features=create_initial_features(observation=self.robot.observation_features),
                    use_videos=self.config.use_videos,
                ),
            )

            self.dataset = LeRobotDataset.create(
                repo_id=self.config.dataset_repo_id,
                fps=self.config.fps,
                features=dataset_features,
                robot_type=self.config.robot_type,
                use_videos=self.config.use_videos,
            )

            self.is_recording = True
            self.is_episode_active = False
            self.current_episode_frames = 0
            self.total_episodes_recorded = 0

            return {
                "status": "started",
                "dataset_repo_id": self.config.dataset_repo_id,
                "fps": self.config.fps,
            }

        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start recording session: {e}") from e

    async def stop(self) -> dict[str, Any]:
        """Stop recording and cleanup resources."""
        if not self.is_recording:
            return {"status": "not_active"}

        self._stop_requested = True

        if self._recording_task and not self._recording_task.done():
            self._recording_task.cancel()
            try:
                await self._recording_task
            except asyncio.CancelledError:
                pass

        if self.is_episode_active:
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

        if self.is_episode_active:
            raise RuntimeError("Episode already active")

        self.is_episode_active = True
        self.current_episode_frames = 0
        self._stop_requested = False

        self._recording_task = asyncio.create_task(self._record_episode_loop())

        return {
            "status": "episode_started",
            "episode_index": self.dataset.num_episodes,
        }

    async def stop_episode(self) -> dict[str, Any]:
        """Stop the current episode without saving."""
        if not self.is_episode_active:
            return {"status": "no_active_episode"}

        self._stop_requested = True

        if self._recording_task and not self._recording_task.done():
            self._recording_task.cancel()
            try:
                await self._recording_task
            except asyncio.CancelledError:
                pass

        self.is_episode_active = False

        return {
            "status": "episode_stopped",
            "frames_recorded": self.current_episode_frames,
        }

    async def save_episode(self) -> dict[str, Any]:
        """Save the current episode to the dataset."""
        if not self.is_episode_active:
            raise RuntimeError("No active episode to save")

        await self.stop_episode()

        try:
            self.dataset.save_episode()
            self.total_episodes_recorded += 1

            return {
                "status": "episode_saved",
                "episode_index": self.dataset.num_episodes - 1,
                "frames": self.current_episode_frames,
                "total_episodes": self.total_episodes_recorded,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to save episode: {e}") from e

    async def discard_episode(self) -> dict[str, Any]:
        """Discard the current episode without saving."""
        if not self.is_episode_active:
            raise RuntimeError("No active episode to discard")

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
            "dataset_total_episodes": self.dataset.num_episodes if self.dataset else 0,
            "dataset_total_frames": self.dataset.meta.total_frames if self.dataset else 0,
        }

    async def _record_episode_loop(self):
        """Internal loop for recording frames during an episode."""
        if not self.robot or not self.dataset:
            raise RuntimeError("Robot or dataset not initialized")

        dt = 1.0 / self.config.fps

        try:
            while not self._stop_requested:
                start_time = time.perf_counter()

                obs = self.robot.get_observation()

                if self.robot and hasattr(self.robot, "cameras"):
                    for key, value in obs.items():
                        if key in self.robot.cameras:
                            self._latest_frame[key] = value

                obs_processed = self.robot_observation_processor(obs)

                observation_frame = build_dataset_frame(self.dataset.features, obs_processed, prefix=OBS_STR)

                if self.teleop:
                    act = self.teleop.get_action()
                    act_processed_teleop = self.teleop_action_processor((act, obs))
                    action_values = act_processed_teleop
                    robot_action_to_send = self.robot_action_processor((act_processed_teleop, obs))
                else:
                    raise RuntimeError("No teleop device available")

                self.robot.send_action(robot_action_to_send)

                action_frame = build_dataset_frame(self.dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": self.config.dataset_task}
                self.dataset.add_frame(frame)

                self.current_episode_frames += 1

                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logging.info("Episode recording loop cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in recording loop: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
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
