from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Iterable, Iterator, Tuple

import jax
import numpy as np

from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .processor import Processor


@dataclass(frozen=True)
class Job:
    """A unit of work for the pipeline.

    ``key`` is opaque to the runner; the caller uses it to associate outputs
    with inputs (e.g. a direction index).
    """
    key: Any
    indices: np.ndarray
    direction_matrix: np.ndarray


_END = object()


class _StageError:
    def __init__(self, exc: BaseException):
        self.exc = exc


class PipelinedRunner:
    """Runs ``processor`` over a stream of ``Job``s with overlapped stages.

    Three concurrent stages connected by bounded queues:
        1. Reader thread: ``processor.prepare`` (host I/O, CPU).
        2. Main thread: ``processor.fit_transform`` (H2D + device dispatch).
        3. Writer thread: ``jax.device_get`` (D2H).

    JAX dispatch is asynchronous, so the main thread can submit iteration
    *i+1*'s device work while the previous result is still being copied back
    by the writer thread.
    """

    def __init__(
        self,
        loader: DataLoader,
        preprocessor: Preprocessor,
        processor: Processor,
        prefetch: int = 2,
    ):
        if prefetch < 1:
            raise ValueError('prefetch must be >= 1')
        self._loader = loader
        self._preprocessor = preprocessor
        self._processor = processor
        self._prefetch = prefetch

    def run(self, jobs: Iterable[Job]) -> Iterator[Tuple[Job, np.ndarray]]:
        prepared_queue: Queue = Queue(maxsize=self._prefetch)
        result_queue: Queue = Queue(maxsize=self._prefetch)
        output_queue: Queue = Queue(maxsize=self._prefetch)

        reader = Thread(
            target=self._reader_loop,
            args=(jobs, prepared_queue),
            daemon=True,
        )
        writer = Thread(
            target=self._writer_loop,
            args=(result_queue, output_queue),
            daemon=True,
        )
        reader.start()
        writer.start()

        # Main thread drives the device-dispatch stage.
        dispatch_thread = Thread(
            target=self._dispatch_loop,
            args=(prepared_queue, result_queue),
            daemon=True,
        )
        dispatch_thread.start()

        try:
            while True:
                item = output_queue.get()
                if item is _END:
                    break
                if isinstance(item, _StageError):
                    raise item.exc
                yield item
        finally:
            reader.join()
            dispatch_thread.join()
            writer.join()

    def _reader_loop(self, jobs: Iterable[Job], out: Queue) -> None:
        try:
            for job in jobs:
                prepared = self._loader.load(
                    job.indices,
                    job.direction_matrix
                )
                out.put((job, prepared))
        except BaseException as e:
            out.put(_StageError(e))
            return
        out.put(_END)

    def _dispatch_loop(self, in_q: Queue, out_q: Queue) -> None:
        try:
            while True:
                item = in_q.get()
                if item is _END:
                    break
                if isinstance(item, _StageError):
                    out_q.put(item)
                    return

                job, loaded = item
                x = self._preprocessor.process(loaded)
                deferred = self._processor.fit_transform(
                    images=x.images_ft, 
                    ctfs=x.ctfs, 
                    count=x.valid_count
                )
                out_q.put((job, deferred))
                
        except BaseException as e:
            out_q.put(_StageError(e))
            return
        out_q.put(_END)

    def _writer_loop(self, in_q: Queue, out_q: Queue) -> None:
        try:
            while True:
                item = in_q.get()
                if item is _END:
                    break
                if isinstance(item, _StageError):
                    out_q.put(item)
                    return
                job, deferred = item
                host_array = jax.device_get(deferred)
                out_q.put((job, host_array))
        except BaseException as e:
            out_q.put(_StageError(e))
            return
        out_q.put(_END)
