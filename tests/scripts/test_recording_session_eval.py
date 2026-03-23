from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_import_stubs() -> None:
    def _register(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    class _ChoiceBase:
        @classmethod
        def get_choice_class(cls, _choice_name):
            return cls

    class _FakeDatasetForImport:
        def __init__(self, *args, **kwargs):
            self.num_episodes = 0
            self.meta = SimpleNamespace(total_frames=0, metadata_buffer_size=1)
            self.features = {}

        @classmethod
        def create(cls, *args, **kwargs):
            return cls()

        def start_image_writer(self, *args, **kwargs):
            return None

        def save_episode(self):
            self.num_episodes += 1

        def clear_episode_buffer(self):
            return None

        def finalize(self):
            return None

    class _FakeRobot:
        robot_type = "mock_robot"

        def __init__(self, *args, **kwargs):
            self.cameras = {}
            self.action_features = {}
            self.observation_features = {}

        def connect(self):
            return None

        def disconnect(self):
            return None

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

    class _FakeTeleoperator:
        def connect(self):
            return None

        def disconnect(self):
            return None

        def get_action(self):
            return {}

    class _FakeCameraConfig(_ChoiceBase):
        @classmethod
        def get_choice_class(cls, _choice_name):
            return cls

        def __init__(self, *args, **kwargs):
            self.color_mode = getattr(_FakeColorMode, "RGB", "RGB")

    class _FakeColorMode:
        RGB = "RGB"
        BGR = "BGR"

    class _FakePreTrainedConfig:
        def __init__(self, *args, **kwargs):
            self.pretrained_path = kwargs.get("pretrained_path")
            self.device = kwargs.get("device", "cpu")
            self.use_amp = kwargs.get("use_amp", False)
            self.type = kwargs.get("type", "fake")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeRobotProcessorPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, value):
            return value

        def reset(self):
            return None

    def _identity(*args, **kwargs):
        return {}

    _register("lerobot.configs.policies", PreTrainedConfig=_FakePreTrainedConfig)
    _register("lerobot.datasets.lerobot_dataset", LeRobotDataset=_FakeDatasetForImport)
    _register(
        "lerobot.datasets.pipeline_features",
        aggregate_pipeline_dataset_features=lambda *args, **kwargs: {},
        create_initial_features=lambda **kwargs: {},
    )
    _register(
        "lerobot.datasets.feature_utils",
        build_dataset_frame=lambda features, data, prefix: {},
        combine_feature_dicts=lambda *dicts: {},
    )
    _register("lerobot.datasets.io_utils", load_info=lambda *args, **kwargs: {})
    _register(
        "lerobot.policies.factory",
        make_policy=lambda *args, **kwargs: SimpleNamespace(
            config=SimpleNamespace(device="cpu", use_amp=False), reset=lambda: None
        ),
        make_pre_post_processors=lambda *args, **kwargs: (
            _FakeRobotProcessorPipeline(),
            _FakeRobotProcessorPipeline(),
        ),
    )
    _register("lerobot.policies.utils", make_robot_action=lambda action, features: action)
    _register(
        "lerobot.processor",
        RobotProcessorPipeline=_FakeRobotProcessorPipeline,
        make_default_processors=lambda: (
            _FakeRobotProcessorPipeline(),
            _FakeRobotProcessorPipeline(),
            _FakeRobotProcessorPipeline(),
        ),
    )
    _register("lerobot.processor.rename_processor", rename_stats=lambda stats, rename_map: stats)
    _register("lerobot.robots", RobotConfig=_ChoiceBase, make_robot_from_config=lambda config: _FakeRobot())
    _register("lerobot.robots.robot", Robot=_FakeRobot)
    _register(
        "lerobot.teleoperators",
        TeleoperatorConfig=_ChoiceBase,
        make_teleoperator_from_config=lambda config: _FakeTeleoperator(),
    )
    _register("lerobot.teleoperators.teleoperator", Teleoperator=_FakeTeleoperator)
    _register(
        "lerobot.cameras.configs",
        CameraConfig=_FakeCameraConfig,
        ColorMode=_FakeColorMode,
    )
    _register(
        "lerobot.utils.control_utils",
        predict_action=lambda **kwargs: {},
        sanity_check_dataset_name=lambda *args, **kwargs: None,
        sanity_check_dataset_robot_compatibility=lambda *args, **kwargs: None,
    )
    _register("lerobot.utils.constants", ACTION="action", HF_LEROBOT_HOME=Path("/tmp"), OBS_STR="observation")
    _register("lerobot.utils.device_utils", get_safe_torch_device=lambda *args, **kwargs: SimpleNamespace(type="cpu"))


def _load_recording_session_module():
    _install_import_stubs()
    module_path = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/recording_session.py"
    spec = importlib.util.spec_from_file_location("recording_session_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recording_session_module = _load_recording_session_module()
EvalConfig = recording_session_module.EvalConfig
EvalSession = recording_session_module.EvalSession


class _FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(device="cpu", use_amp=False)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _FakePipeline:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _FakeDataset:
    def __init__(self, num_episodes: int = 4):
        self.num_episodes = num_episodes
        self.save_episode_calls = 0
        self.clear_episode_buffer_calls = 0
        self.meta = SimpleNamespace(total_frames=120, metadata_buffer_size=1)
        self.features = {}

    def save_episode(self):
        self.save_episode_calls += 1
        self.num_episodes += 1

    def clear_episode_buffer(self):
        self.clear_episode_buffer_calls += 1


def _make_config() -> EvalConfig:
    return EvalConfig(
        robot_type="so101_follower",
        robot_port="/dev/ttyACM0",
        robot_id="follower",
        robot_cameras={},
        dataset_repo_id="test-user/eval_so101",
        dataset_task="pick the cube",
        policy_path="test-user/model",
        num_episodes=5,
    )


def test_eval_session_resets_policy_and_processors_each_episode():
    session = EvalSession(_make_config())
    session.is_recording = True
    session.dataset = _FakeDataset()
    session.policy = _FakePolicy()
    session.preprocessor = _FakePipeline()
    session.postprocessor = _FakePipeline()
    session.current_episode_frames = 8

    start_result = asyncio.run(session.start_episode())
    assert start_result == {"status": "episode_started", "episode_index": 4}
    assert session.current_episode_frames == 0
    assert session.policy.reset_calls == 1
    assert session.preprocessor.reset_calls == 1
    assert session.postprocessor.reset_calls == 1

    session.is_episode_active = True
    session.current_episode_frames = 6
    save_result = asyncio.run(session.save_episode())
    assert save_result["status"] == "episode_saved"
    assert save_result["frames"] == 6
    assert save_result["episode_index"] == 4
    assert save_result["total_episodes"] == 1
    assert session.dataset.save_episode_calls == 1
    assert session.dataset.num_episodes == 5
    assert session.is_episode_active is False

    session.is_episode_active = True
    session.current_episode_frames = 3
    discard_result = asyncio.run(session.discard_episode())
    assert discard_result == {"status": "episode_discarded", "frames_discarded": 3}
    assert session.dataset.clear_episode_buffer_calls == 1
    assert session.is_episode_active is False
