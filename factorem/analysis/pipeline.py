from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import jax
import numpy as np
import vitrio as vio

from .preprocessor import Preprocessor
from .processor import Processor


@dataclass(frozen=True)
class Job:
    """The particles of one projection direction, analyzed together."""
    key: int
    indices: np.ndarray
    direction_matrix: np.ndarray


def make_dataset(
    image_names: Sequence[str],
    rotations: np.ndarray,
    shifts: np.ndarray,
    defocus: np.ndarray,
    image_prefix: Optional[str] = None,
) -> vio.data.Dataset:
    """Describe every particle: its image and the metadata to align it."""
    locations = vio.data.parse_locations(image_names, root=image_prefix)
    return vio.data.Dataset(
        {
            'image': vio.data.ImageSource(locations, data_type=np.float32),
            'rotation': rotations.astype(np.float32),
            'shift': shifts.astype(np.float32),
            'defocus': defocus.astype(np.float32),
        },
        index=None,
    )


def _pad_to_multiple(indices: np.ndarray, grain_size: int) -> np.ndarray:
    """Repeat rows so that batch sizes take few distinct values, which keeps
    JAX from compiling again for every group. Padded rows are ignored by the
    processors, which only look at the first ``len(indices)`` rows."""
    padded_count = -(-len(indices) // grain_size) * grain_size
    return np.resize(indices, padded_count)


def analyze_directions(
    dataset: vio.data.Dataset,
    jobs: Sequence[Job],
    preprocessor: Preprocessor,
    processor: Processor,
    device: jax.Device,
    grain_size: int = 256,
    prefetch: int = 4,
) -> Iterator[Tuple[Job, np.ndarray]]:
    """Embed the particles of each job, yielding results in job order.

    Stages overlap without threads of our own: vitrio reads the next
    ``prefetch`` groups in the background and copies each one to ``device``,
    JAX dispatches the computation asynchronously, and each result is copied
    back to the host while the next group is computed.
    """
    batches = vio.data.DataLoader(
        dataset,
        batch_sampler=[_pad_to_multiple(job.indices, grain_size) for job in jobs],
        output='jax',
        device=device,
        prefetch=prefetch,
    )

    previous = None
    with jax.default_device(device):
        for job, batch in zip(jobs, batches):
            x = preprocessor(batch, job.direction_matrix)
            y = processor.fit_transform(x.images_ft, x.ctfs, len(job.indices))
            y.copy_to_host_async()

            # Wait for the previous result only once this one is dispatched.
            if previous is not None:
                yield previous[0], np.asarray(previous[1])
            previous = (job, y)

        if previous is not None:
            yield previous[0], np.asarray(previous[1])
