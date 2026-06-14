import glob
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
  from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Architecture short name (no underscores). Imported by callers via `from .image_video_dataset import ARCHITECTURE_WAN`.
ARCHITECTURE_WAN = "wan"

from musubi_tuner.dataset.media_utils import VIDEO_EXTENSIONS, glob_videos, load_video, resize_image_to_bucket


class ItemInfo:
  def __init__(
    self,
    item_key: str,
    caption: str,
    original_size: tuple[int, int],
    bucket_size: tuple[Any] | None = None,
    frame_count: int | None = None,
    content: np.ndarray | list[np.ndarray] | None = None,
    latent_cache_path: str | None = None,
  ) -> None:
    self.item_key = item_key
    self.caption = caption
    self.original_size = original_size
    self.bucket_size = bucket_size
    self.frame_count = frame_count
    self.content = content
    self.latent_cache_path = latent_cache_path
    self.text_encoder_output_cache_path: str | None = None

    # np.ndarray for video, list[np.ndarray] for image with multiple controls
    self.control_content: np.ndarray | list[np.ndarray] | None = None


  def __str__(self) -> str:
    return (
      f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
      + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
      + f"frame_count={self.frame_count}, latent_cache_path={self.latent_cache_path}, "
      + f"content={[c.shape for c in self.content] if isinstance(self.content, list) else (self.content.shape if self.content is not None else None)}), "
      + f"control_content={[cc.shape for cc in self.control_content] if isinstance(self.control_content, list) else (self.control_content.shape if self.control_content is not None else None)})"
    )


from safetensors.torch import save_file as safetensors_save_file

from musubi_tuner.dataset.bucket import BucketBatchManager, BucketSelector
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str


class ContentDatasource:
  def __init__(self):
    self.caption_only = False
    self.has_control = False

  def set_caption_only(self, caption_only: bool):
    self.caption_only = caption_only

  def is_indexable(self):
    return False

  def get_caption(self, idx: int) -> tuple[str, str]:
    raise NotImplementedError

  def __len__(self):
    raise NotImplementedError

  def __iter__(self):
    raise NotImplementedError

  def __next__(self):
    raise NotImplementedError


class VideoDatasource(ContentDatasource):
  def __init__(self):
    super().__init__()
    self.start_frame = None
    self.end_frame = None
    self.bucket_selector = None
    self.source_fps = None
    self.target_fps = None

  def __len__(self):
    raise NotImplementedError

  def get_video_data_from_path(
    self,
    video_path: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    bucket_selector: "BucketSelector | None" = None,
  ) -> list[np.ndarray]:
    start_frame = start_frame if start_frame is not None else self.start_frame
    end_frame = end_frame if end_frame is not None else self.end_frame
    bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector
    return load_video(video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps)

  def get_control_data_from_path(
    self,
    control_path: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
    bucket_selector: "BucketSelector | None" = None,
  ) -> list[np.ndarray]:
    start_frame = start_frame if start_frame is not None else self.start_frame
    end_frame = end_frame if end_frame is not None else self.end_frame
    bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector
    return load_video(control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps)

  def set_start_and_end_frame(self, start_frame: int | None, end_frame: int | None):
    self.start_frame = start_frame
    self.end_frame = end_frame

  def set_bucket_selector(self, bucket_selector: "BucketSelector"):
    self.bucket_selector = bucket_selector

  def set_source_and_target_fps(self, source_fps: float | None, target_fps: float | None):
    self.source_fps = source_fps
    self.target_fps = target_fps

  def __iter__(self):
    raise NotImplementedError

  def __next__(self):
    raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
  def __init__(self, video_directory: str, caption_extension: str | None = None, control_directory: str | None = None):
    super().__init__()
    self.video_directory = video_directory
    self.caption_extension = caption_extension
    self.control_directory = control_directory
    self.current_idx = 0

    logger.info(f"glob videos in {self.video_directory}")
    self.video_paths = glob_videos(self.video_directory)
    logger.info(f"found {len(self.video_paths)} videos")

    if self.control_directory is not None:
      logger.info(f"glob control videos in {self.control_directory}")
      self.has_control = True
      self.control_paths = {}
      for video_path in self.video_paths:
        video_basename = os.path.basename(video_path)
        control_path = os.path.join(self.control_directory, video_basename)
        if os.path.exists(control_path):
          self.control_paths[video_path] = control_path
        else:
          base_name = os.path.splitext(video_basename)[0]
          potential_path = os.path.join(self.control_directory, base_name)
          if os.path.isdir(potential_path):
            self.control_paths[video_path] = potential_path
          else:
            for ext in VIDEO_EXTENSIONS:
              potential_path = os.path.join(self.control_directory, base_name + ext)
              if os.path.exists(potential_path):
                self.control_paths[video_path] = potential_path
                break

      logger.info(f"found {len(self.control_paths)} matching control videos/images")
      missing_controls = len(self.video_paths) - len(self.control_paths)
      if missing_controls > 0:
        missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
        raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}")

  def is_indexable(self):
    return True

  def __len__(self):
    return len(self.video_paths)

  def get_video_data(
    self,
    idx: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
    bucket_selector: "BucketSelector | None" = None,
  ) -> tuple[str, list[np.ndarray], str, list[np.ndarray] | None]:
    video_path = self.video_paths[idx]
    video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)
    _, caption = self.get_caption(idx)
    control = None
    if self.control_directory is not None and video_path in self.control_paths:
      control_path = self.control_paths[video_path]
      control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)
    return video_path, video, caption, control

  def get_caption(self, idx: int) -> tuple[str, str]:
    video_path = self.video_paths[idx]
    caption_path = os.path.splitext(video_path)[0] + self.caption_extension if self.caption_extension else ""
    with open(caption_path, encoding="utf-8") as f:
      caption = f.read().strip()
    return video_path, caption

  def __iter__(self):
    self.current_idx = 0
    return self

  def __next__(self):
    if self.current_idx >= len(self.video_paths):
      raise StopIteration

    if self.caption_only:
      def create_caption_fetcher(index):
        return lambda: self.get_caption(index)
      fetcher = create_caption_fetcher(self.current_idx)
    else:
      def create_fetcher(index):
        return lambda: self.get_video_data(index)
      fetcher = create_fetcher(self.current_idx)

    self.current_idx += 1
    return fetcher


class VideoJsonlDatasource(VideoDatasource):
  def __init__(self, video_jsonl_file: str):
    super().__init__()
    self.video_jsonl_file = video_jsonl_file
    self.current_idx = 0

    logger.info(f"load video jsonl from {self.video_jsonl_file}")
    self.data = []
    with open(self.video_jsonl_file, encoding="utf-8") as f:
      for line in f:
        data = json.loads(line)
        self.data.append(data)
    logger.info(f"loaded {len(self.data)} videos")

    self.has_control = any("control_path" in item for item in self.data)
    if self.has_control:
      control_count = sum(1 for item in self.data if "control_path" in item)
      if control_count < len(self.data):
        missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
        raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
      logger.info(f"found {control_count} control videos/images in JSONL data")

  def is_indexable(self):
    return True

  def __len__(self):
    return len(self.data)

  def get_video_data(
    self,
    idx: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
    bucket_selector: "BucketSelector | None" = None,
  ) -> tuple[str, list[np.ndarray], str, list[np.ndarray] | None]:
    data = self.data[idx]
    video_path = data["video_path"]
    video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)
    caption = data["caption"]
    control = None
    if "control_path" in data and data["control_path"]:
      control_path = data["control_path"]
      control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)
    return video_path, video, caption, control

  def get_caption(self, idx: int) -> tuple[str, str]:
    data = self.data[idx]
    video_path = data["video_path"]
    caption = data["caption"]
    return video_path, caption

  def __iter__(self):
    self.current_idx = 0
    return self

  def __next__(self):
    if self.current_idx >= len(self.data):
      raise StopIteration

    if self.caption_only:
      def create_caption_fetcher(index):
        return lambda: self.get_caption(index)
      fetcher = create_caption_fetcher(self.current_idx)
    else:
      def create_fetcher(index):
        return lambda: self.get_video_data(index)
      fetcher = create_fetcher(self.current_idx)

    self.current_idx += 1
    return fetcher


class BaseDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    resolution: tuple[int, int] = (960, 544),
    caption_extension: str | None = None,
    batch_size: int = 1,
    num_repeats: int = 1,
    cache_directory: str | None = None,
    debug_dataset: bool = False,
  ):
    self.resolution = resolution
    self.caption_extension = caption_extension
    self.batch_size = batch_size
    self.num_repeats = num_repeats
    self.cache_directory = cache_directory
    self.debug_dataset = debug_dataset
    self.seed = None
    self.current_epoch = 0
    self.shared_epoch = None

  def get_metadata(self) -> dict:
    metadata = {
      "resolution": self.resolution,
      "caption_extension": self.caption_extension,
      "batch_size_per_device": self.batch_size,
      "num_repeats": self.num_repeats,
    }
    return metadata

  def get_all_latent_cache_files(self):
    return glob.glob(os.path.join(self.cache_directory, "*_wan.safetensors"))

  def get_all_text_encoder_output_cache_files(self):
    return glob.glob(os.path.join(self.cache_directory, "*_wan_te.safetensors"))

  def get_latent_cache_path(self, item_info: ItemInfo) -> str:
    """
    Returns the cache path for the latent tensor.

    item_info: ItemInfo object

    Returns:
        str: cache path

    cache_path is based on the item_key and the resolution.
    """
    w, h = item_info.original_size
    basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
    assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
    return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_wan.safetensors")

  def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
    basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
    assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
    return os.path.join(self.cache_directory, f"{basename}_wan_te.safetensors")

  def retrieve_latent_cache_batches(self, num_workers: int):
    raise NotImplementedError

  def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
    raise NotImplementedError

  def prepare_for_training(self, num_timestep_buckets: int | None = None):
    pass

  def set_seed(self, seed: int, shared_epoch: SharedEpoch):
    self.seed = seed
    self.shared_epoch = shared_epoch

  def set_current_epoch(self, epoch):
    assert self.shared_epoch is not None, "shared_epoch is None"
    assert self.shared_epoch.value == epoch, "shared_epoch does not match"

  def set_max_train_steps(self, max_train_steps):
    self.max_train_steps = max_train_steps

  def shuffle_buckets(self):
    raise NotImplementedError

  def __len__(self):
    return NotImplementedError

  def __getitem__(self, idx):
    assert self.shared_epoch is not None, "shared_epoch is None"
    epoch = self.shared_epoch.value
    if epoch > self.current_epoch:
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is None or worker_info.id == 0:
        logger.info(f"epoch is incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
      num_epochs = epoch - self.current_epoch
      for _ in range(num_epochs):
        self.current_epoch += 1
        self.shuffle_buckets()
    elif epoch < self.current_epoch:
      logger.warning(f"epoch is not incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
      self.current_epoch = epoch

  def _default_retrieve_text_encoder_output_cache_batches(self, datasource: ContentDatasource, batch_size: int, num_workers: int):
    datasource.set_caption_only(True)
    executor = ThreadPoolExecutor(max_workers=num_workers)

    data: list[ItemInfo] = []
    futures = []

    def aggregate_future(consume_all: bool = False):
      while len(futures) >= num_workers or (consume_all and len(futures) > 0):
        completed_futures = [future for future in futures if future.done()]
        if len(completed_futures) == 0:
          if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
            time.sleep(0.1)
            continue
          else:
            break  # submit batch if possible

        for future in completed_futures:
          item_key, caption = future.result()
          item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
          item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
          data.append(item_info)

          futures.remove(future)

    def submit_batch(flush: bool = False):
      nonlocal data
      if len(data) >= batch_size or (len(data) > 0 and flush):
        batch = data[0:batch_size]
        if len(data) > batch_size:
          data = data[batch_size:]
        else:
          data = []
        return batch
      return None

    for fetch_op in datasource:
      future = executor.submit(fetch_op)
      futures.append(future)
      aggregate_future()
      while True:
        batch = submit_batch()
        if batch is None:
          break
        yield batch

    aggregate_future(consume_all=True)
    while True:
      batch = submit_batch(flush=True)
      if batch is None:
        break
      yield batch

    executor.shutdown()


class VideoDataset(BaseDataset):
  TARGET_FPS_WAN = 16.0

  def __init__(
    self,
    resolution: tuple[int, int],
    caption_extension: str | None,
    batch_size: int,
    num_repeats: int,
    target_frames: list[int] | None = None,
    source_fps: float | None = None,
    video_directory: str | None = None,
    video_jsonl_file: str | None = None,
    control_directory: str | None = None,
    cache_directory: str | None = None,
    debug_dataset: bool = False,
  ):
    super().__init__(
      resolution,
      caption_extension,
      batch_size,
      num_repeats,
      cache_directory,
      debug_dataset,
    )
    self.video_directory = video_directory
    self.video_jsonl_file = video_jsonl_file
    self.control_directory = control_directory
    self.source_fps = source_fps

    self.vae_frame_stride = 4
    self.target_fps = VideoDataset.TARGET_FPS_WAN

    if target_frames is not None:
      target_frames = list(set(target_frames))
      target_frames.sort()

      # round each value to N*4+1 (VAE temporal stride requirement)
      rounded_target_frames = [(f - 1) // self.vae_frame_stride * self.vae_frame_stride + 1 for f in target_frames]
      rounded_target_frames = list(set(rounded_target_frames))
      rounded_target_frames.sort()

      if target_frames != rounded_target_frames:
        logger.warning(f"target_frames are rounded to {rounded_target_frames}")

      target_frames = tuple(rounded_target_frames)

    self.target_frames = target_frames

    if video_directory is not None:
      self.datasource = VideoDirectoryDatasource(video_directory, caption_extension, control_directory)
    elif video_jsonl_file is not None:
      self.datasource = VideoJsonlDatasource(video_jsonl_file)

    # None end_frame loads all frames; used when target_frames is auto-computed per video
    self.datasource.set_start_and_end_frame(0, max(self.target_frames) if self.target_frames is not None else None)

    if self.cache_directory is None:
      self.cache_directory = self.video_directory

    self.batch_manager = None
    self.num_train_items = 0
    self.has_control = self.datasource.has_control

  def get_metadata(self):
    metadata = super().get_metadata()
    if self.video_directory is not None:
      metadata["video_directory"] = os.path.basename(self.video_directory)
    if self.video_jsonl_file is not None:
      metadata["video_jsonl_file"] = os.path.basename(self.video_jsonl_file)
    if self.control_directory is not None:
      metadata["control_directory"] = os.path.basename(self.control_directory)
    metadata["target_frames"] = self.target_frames
    metadata["source_fps"] = self.source_fps
    metadata["has_control"] = self.has_control
    return metadata

  def retrieve_latent_cache_batches(self, num_workers: int):
    buckset_selector = BucketSelector(self.resolution)
    self.datasource.set_bucket_selector(buckset_selector)
    if self.source_fps is not None:
      self.datasource.set_source_and_target_fps(self.source_fps, self.target_fps)
    else:
      self.datasource.set_source_and_target_fps(None, None)  # no conversion

    executor = ThreadPoolExecutor(max_workers=num_workers)

    # key: (width, height, frame_count) and optional latent_window_size, value: [ItemInfo]
    batches: dict[tuple[Any], list[ItemInfo]] = {}
    futures = []

    def aggregate_future(consume_all: bool = False):
      while len(futures) >= num_workers or (consume_all and len(futures) > 0):
        completed_futures = [future for future in futures if future.done()]
        if len(completed_futures) == 0:
          if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
            time.sleep(0.1)
            continue
          else:
            break  # submit batch if possible

        for future in completed_futures:
          original_frame_size, video_key, video, caption, control = future.result()

          frame_count = len(video)
          video = np.stack(video, axis=0)
          height, width = video.shape[1:3]
          bucket_reso = (width, height)  # already resized

          # process control images if available
          control_video = None
          if control is not None:
            # set frame count to the same as video
            if len(control) > frame_count:
              control = control[:frame_count]
            elif len(control) < frame_count:
              # if control is shorter than video, repeat the last frame
              last_frame = control[-1]
              control.extend([last_frame] * (frame_count - len(control)))
            control_video = np.stack(control, axis=0)

          crop_pos_and_frames = []
          if self.target_frames is None:
            # snap down to largest valid VAE frame count for this video
            auto_frames = (frame_count - 1) // self.vae_frame_stride * self.vae_frame_stride + 1
            if auto_frames >= 1:
              crop_pos_and_frames.append((0, auto_frames))
          else:
            for target_frame in self.target_frames:
              if frame_count >= target_frame:
                crop_pos_and_frames.append((0, target_frame))

          for crop_pos, target_frame in crop_pos_and_frames:
            cropped_video = video[crop_pos : crop_pos + target_frame]
            body, ext = os.path.splitext(video_key)
            item_key = f"{body}_{crop_pos:05d}-{target_frame:03d}{ext}"
            batch_key = (*bucket_reso, target_frame)  # bucket_reso with frame_count

            # crop control video if available
            cropped_control = None
            if control_video is not None:
              cropped_control = control_video[crop_pos : crop_pos + target_frame]

            item_info = ItemInfo(item_key, caption, original_frame_size, batch_key, frame_count=target_frame, content=cropped_video)
            item_info.latent_cache_path = self.get_latent_cache_path(item_info)
            item_info.control_content = cropped_control  # None is allowed

            batch = batches.get(batch_key, [])
            batch.append(item_info)
            batches[batch_key] = batch

          futures.remove(future)

    def submit_batch(flush: bool = False):
      for key in batches:
        if len(batches[key]) >= self.batch_size or flush:
          batch = batches[key][0 : self.batch_size]
          if len(batches[key]) > self.batch_size:
            batches[key] = batches[key][self.batch_size :]
          else:
            del batches[key]
          return key, batch
      return None, None

    for operator in self.datasource:

      def fetch_and_resize(op: Callable[..., Any]) -> tuple[tuple[int, int], str, list[np.ndarray], str, list[np.ndarray] | None]:
        video_key, video, caption, control = op()

        video: list[np.ndarray]
        frame_size = (video[0].shape[1], video[0].shape[0])

        # resize if necessary
        bucket_reso = buckset_selector.get_bucket_resolution(frame_size)
        video = [resize_image_to_bucket(frame, bucket_reso) for frame in video]

        # resize control if necessary
        if control is not None:
          control = [resize_image_to_bucket(frame, bucket_reso) for frame in control]

        return frame_size, video_key, video, caption, control

      future = executor.submit(fetch_and_resize, operator)
      futures.append(future)
      aggregate_future()
      while True:
        key, batch = submit_batch()
        if key is None:
          break
        yield key, batch

    aggregate_future(consume_all=True)
    while True:
      key, batch = submit_batch(flush=True)
      if key is None:
        break
      yield key, batch

    executor.shutdown()

  def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
    return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

  def prepare_for_training(self, num_timestep_buckets: int | None = None):
    bucket_selector = BucketSelector(self.resolution)

    # glob cache files
    latent_cache_files = glob.glob(os.path.join(self.cache_directory, "*_wan.safetensors"))

    # assign cache files to item info
    bucketed_item_info: dict[tuple[int, int, int], list[ItemInfo]] = {}  # (width, height, frame_count) -> [ItemInfo]
    for cache_file in latent_cache_files:
      tokens = os.path.basename(cache_file).split("_")

      image_size = tokens[-2]  # 0000x0000
      image_width, image_height = map(int, image_size.split("x"))
      image_size = (image_width, image_height)

      frame_pos, frame_count = tokens[-3].split("-")[:2]  # "00000-000", or optional section index "00000-000-00"
      frame_pos, frame_count = int(frame_pos), int(frame_count)

      item_key = "_".join(tokens[:-3])
      text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_wan_te.safetensors")
      if not os.path.exists(text_encoder_output_cache_file):
        logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
        continue

      bucket_reso = bucket_selector.get_bucket_resolution(image_size)
      bucket_reso = (*bucket_reso, frame_count)
      item_info = ItemInfo(item_key, "", image_size, bucket_reso, frame_count=frame_count, latent_cache_path=cache_file)
      item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

      bucket = bucketed_item_info.get(bucket_reso, [])
      for _ in range(self.num_repeats):
        bucket.append(item_info)
      bucketed_item_info[bucket_reso] = bucket

    # prepare batch manager
    self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, num_timestep_buckets=num_timestep_buckets)
    self.batch_manager.show_bucket_info()

    self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

  def shuffle_buckets(self):
    # set random seed for this epoch
    random.seed(self.seed + self.current_epoch)
    self.batch_manager.shuffle()

  def __len__(self):
    if self.batch_manager is None:
      return 100  # dummy value
    return len(self.batch_manager)

  def __getitem__(self, idx):
    super().__getitem__(idx)
    return self.batch_manager[idx]


class DatasetGroup(torch.utils.data.ConcatDataset):
  def __init__(self, datasets: Sequence[VideoDataset]):
    super().__init__(datasets)
    self.datasets: list[VideoDataset] = datasets
    self.num_train_items = 0
    for dataset in self.datasets:
      self.num_train_items += dataset.num_train_items

  def set_current_epoch(self, epoch):
    for dataset in self.datasets:
      dataset.set_current_epoch(epoch)

  def set_max_train_steps(self, max_train_steps):
    for dataset in self.datasets:
      dataset.set_max_train_steps(max_train_steps)


def save_latent_cache_wan(
  item_info: ItemInfo,
  latent: torch.Tensor,
  clip_embed: torch.Tensor | None,
  image_latent: torch.Tensor | None,
  control_latent: torch.Tensor | None,
  f_indices: list[int] | None = None,
):
  assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

  _, F, H, W = latent.shape
  dtype_str = dtype_to_str(latent.dtype)
  sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

  if clip_embed is not None:
    sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()
  if image_latent is not None:
    sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()
  if control_latent is not None:
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()
  if f_indices is not None:
    sd[f"f_indices_{dtype_to_str(torch.int32)}"] = torch.tensor(f_indices, dtype=torch.int32)

  metadata = {
    "architecture": ARCHITECTURE_WAN,
    "width": f"{item_info.original_size[0]}",
    "height": f"{item_info.original_size[1]}",
    "format_version": "1.0.1",
  }
  if item_info.frame_count is not None:
    metadata["frame_count"] = f"{item_info.frame_count}"

  for key, value in sd.items():
    if torch.isnan(value).any():
      logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
      value[torch.isnan(value)] = 0

  os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
  safetensors_save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
  dtype_str = dtype_to_str(embed.dtype)
  sd = {f"varlen_t5_{dtype_str}": embed.detach().cpu()}

  for key, value in sd.items():
    if torch.isnan(value).any():
      logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
      value[torch.isnan(value)] = 0

  metadata = {
    "architecture": ARCHITECTURE_WAN,
    "caption1": item_info.caption,
    "format_version": "1.0.1",
  }

  if os.path.exists(item_info.text_encoder_output_cache_path):
    with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
      existing_metadata = f.metadata()
      for key in f.keys():
        if key not in sd:
          sd[key] = f.get_tensor(key)
    assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
    if existing_metadata["caption1"] != metadata["caption1"]:
      logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
    existing_metadata.pop("caption1", None)
    existing_metadata.pop("format_version", None)
    metadata.update(existing_metadata)
  else:
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)

  safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)
