from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
  """
  Output class for the scheduler's `step` function output.

  Args:
      prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
          Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
          denoising loop.
  """

  prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
  """
  Euler scheduler.

  This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
  methods the library implements for all schedulers such as loading and saving.

  Args:
      num_train_timesteps (`int`, defaults to 1000):
          The number of diffusion steps to train the model.
      timestep_spacing (`str`, defaults to `"linspace"`):
          The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
          Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
      shift (`float`, defaults to 1.0):
          The shift value for the timestep schedule.
      reverse (`bool`, defaults to `True`):
          Whether to reverse the timestep schedule.
  """

  _compatibles = []
  order = 1

  @register_to_config
  def __init__(
    self,
    num_train_timesteps: int = 1000,
    shift: float = 1.0,
    reverse: bool = True,
    solver: str = "euler",
  ):
    sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

    if not reverse:
      sigmas = sigmas.flip(0)

    self.sigmas = sigmas
    self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

    self._step_index = None

  @property
  def step_index(self):
    """
    The index counter for current timestep. It will increase 1 after each scheduler step.
    """
    return self._step_index

  def _sigma_to_t(self, sigma):
    return sigma * self.config.num_train_timesteps

  def set_timesteps(
    self,
    num_inference_steps: int,
    device: str | torch.device = None,
  ):
    self.num_inference_steps = num_inference_steps

    sigmas = torch.linspace(1, 0, num_inference_steps + 1)
    sigmas = self.sd3_time_shift(sigmas)

    if not self.config.reverse:
      sigmas = 1 - sigmas

    self.sigmas = sigmas
    self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

    # Reset step index
    self._step_index = None

  def index_for_timestep(self, timestep, schedule_timesteps=None):
    if schedule_timesteps is None:
      schedule_timesteps = self.timesteps

    indices = (schedule_timesteps == timestep).nonzero()

    # The sigma index that is taken for the **very** first `step`
    # is always the second index (or the last index if there is only 1)
    # This way we can ensure we don't accidentally skip a sigma in
    # case we start in the middle of the denoising schedule (e.g. for image-to-image)
    pos = 1 if len(indices) > 1 else 0

    return indices[pos].item()

  def _init_step_index(self, timestep):
    if isinstance(timestep, torch.Tensor):
      timestep = timestep.to(self.timesteps.device)
    self._step_index = self.index_for_timestep(timestep)

  def sd3_time_shift(self, t: torch.Tensor):
    return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

  def step(
    self,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    return_dict: bool = True,
  ) -> FlowMatchDiscreteSchedulerOutput | tuple:
    if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
      raise ValueError(
        "Passing integer indices as timesteps to EulerDiscreteScheduler.step() is not supported. "
        "Pass one of the `scheduler.timesteps` as a timestep."
      )

    if self.step_index is None:
      self._init_step_index(timestep)

    sample = sample.to(torch.float32)
    dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
    prev_sample = sample + model_output.to(torch.float32) * dt

    self._step_index += 1

    if not return_dict:
      return (prev_sample,)

    return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

  def __len__(self):
    return self.config.num_train_timesteps
