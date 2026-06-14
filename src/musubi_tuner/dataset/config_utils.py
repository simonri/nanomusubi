import argparse
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Optional

import toml

if TYPE_CHECKING:
  from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]

import logging

from musubi_tuner.dataset.image_video_dataset import DatasetGroup, VideoDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BaseDatasetParams:
  resolution: tuple[int, int] = (960, 544)
  caption_extension: str | None = None
  batch_size: int = 1
  num_repeats: int = 1
  cache_directory: str | None = None
  debug_dataset: bool = False


@dataclass
class VideoDatasetParams(BaseDatasetParams):
  video_directory: str | None = None
  video_jsonl_file: str | None = None
  control_directory: str | None = None
  target_frames: Sequence[int] | None = None
  source_fps: float | None = None


@dataclass
class DatasetBlueprint:
  params: VideoDatasetParams


@dataclass
class DatasetGroupBlueprint:
  datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
  dataset_group: DatasetGroupBlueprint


def _normalize_resolution(config: dict) -> dict:
  if "resolution" in config:
    r = config["resolution"]
    if isinstance(r, (int, float)):
      config["resolution"] = (int(r), int(r))
    elif isinstance(r, (list, tuple)):
      config["resolution"] = tuple(int(x) for x in r)
  return config


def generate_blueprint(user_config: dict, argparse_namespace: argparse.Namespace) -> Blueprint:
  argparse_config = {k: v for k, v in vars(argparse_namespace).items() if v is not None}
  general_config = _normalize_resolution(dict(user_config.get("general", {})))

  dataset_blueprints = []
  for raw_dataset_config in user_config.get("datasets", []):
    dataset_config = _normalize_resolution(dict(raw_dataset_config))
    params = _build_params(VideoDatasetParams, [dataset_config, general_config, argparse_config])
    dataset_blueprints.append(DatasetBlueprint(params))

  return Blueprint(DatasetGroupBlueprint(dataset_blueprints))


def _build_params(param_klass, fallbacks: list[dict]):
  defaults = asdict(param_klass())
  values = {name: _pick(name, fallbacks, defaults[name]) for name in defaults}
  return param_klass(**values)


def _pick(key: str, fallbacks: list[dict], default=None):
  for d in fallbacks:
    if d.get(key) is not None:
      return d[key]
  return default


def generate_dataset_group_by_blueprint(
  dataset_group_blueprint: DatasetGroupBlueprint,
  training: bool = False,
  num_timestep_buckets: int | None = None,
  shared_epoch: SharedEpoch = None,
) -> DatasetGroup:
  datasets: list[VideoDataset] = []

  for dataset_blueprint in dataset_group_blueprint.datasets:
    dataset = VideoDataset(**asdict(dataset_blueprint.params))
    datasets.append(dataset)

  cache_directories = [dataset.cache_directory for dataset in datasets]
  if len(set(cache_directories)) != len(cache_directories):
    raise ValueError(
      "cache directory should be unique for each dataset"
      + " / cache directory は各データセットごとに異なる必要があります（指定されていない場合はimage/video directoryが使われるので注意）"
    )

  info = ""
  for i, dataset in enumerate(datasets):
    info += dedent(
      f"""\
      [Dataset {i}]
        resolution: {dataset.resolution}
        batch_size: {dataset.batch_size}
        num_repeats: {dataset.num_repeats}
        caption_extension: "{dataset.caption_extension}"
        cache_directory: "{dataset.cache_directory}"
        video_directory: "{dataset.video_directory}"
        video_jsonl_file: "{dataset.video_jsonl_file}"
        control_directory: "{dataset.control_directory}"
        target_frames: {dataset.target_frames}
        source_fps: {dataset.source_fps}
    """
    )
  logger.info(f"{info}")

  seed = random.randint(0, 2**31)
  for i, dataset in enumerate(datasets):
    dataset.set_seed(seed, shared_epoch)
    if training:
      dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)

  return DatasetGroup(datasets)


def load_user_config(file: str) -> dict:
  file: Path = Path(file)
  if not file.is_file():
    raise ValueError(f"file not found / ファイルが見つかりません: {file}")
  try:
    config = toml.load(file)
  except Exception:
    logger.error(
      f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
    )
    raise
  return config
