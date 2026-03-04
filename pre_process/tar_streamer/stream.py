import concurrent.futures
import io
import logging
import tarfile
from itertools import islice
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .models import (
    ConcurrencyConfig,
    ImageRecord,
    OutputFormat,
    StreamConfig,
)

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


class TarImageStream:

    def __init__(self, config: StreamConfig, concurrency: ConcurrencyConfig):
        self._config = config
        self._concurrency = concurrency
        self._tar_file: Optional[tarfile.TarFile] = None
        self._current_tar_path: Optional[Path] = None
        self._n_yielded = 0
        self._n_skipped = 0

    def __repr__(self) -> str:
        state = "open" if self._tar_file else "closed"
        return (
            f"TarImageStream(tars={len(self._config.tar_paths)}, "
            f"state={state}, yielded={self._n_yielded})"
        )

    def __len__(self) -> int:
        total = 0
        for tar_path in self._config.tar_paths:
            with tarfile.open(tar_path, mode="r:*") as tf:
                total += sum(1 for m in tf.getmembers() if self._is_image(m))
        return total

    def __iter__(self) -> Generator[ImageRecord, None, None]:
        yield from self.stream()

    def _is_image(self, member: tarfile.TarInfo) -> bool:
        return member.isfile() and Path(member.name).suffix.lower() in self._config.extensions

    def _check_size(self, image: Image.Image) -> bool:
        w, h = image.size
        min_w, min_h = self._config.min_size
        max_w, max_h = self._config.max_size
        return min_w <= w <= max_w and min_h <= h <= max_h

    def _at_limit(self) -> bool:
        return (
            self._config.max_images is not None
            and self._n_yielded >= self._config.max_images
        )

    def _image_members(
        self, tar_file: tarfile.TarFile
    ) -> Generator[tarfile.TarInfo, None, None]:
        for member in tar_file:
            if self._is_image(member):
                yield member

    def _decode(
        self, member_name: str, raw_bytes: bytes
    ) -> tuple[str, Optional[Image.Image]]:
        try:
            buf = io.BytesIO(raw_bytes)
            image = Image.open(buf)
            image.load()
            if self._config.convert_mode:
                image = image.convert(self._config.convert_mode)
            if not self._check_size(image):
                return member_name, None
            return member_name, image
        except Exception as exc:
            logger.warning("Failed to decode %s: %s", member_name, exc)
            return member_name, None

    def _to_record(
        self, name: str, image: Image.Image, tar_path: str
    ) -> ImageRecord:
        arr = np.asarray(image)
        if self._config.output_format == OutputFormat.NUMPY:
            return ImageRecord(
                image=arr,
                filename=name,
                dtype=arr.dtype,
                shape=arr.shape,
                tar_path=tar_path,
            )
        return ImageRecord(
            image=image,
            filename=name,
            dtype=arr.dtype,
            shape=arr.shape,
            tar_path=tar_path,
        )

    def _stream_sequential(
        self, tar_file: tarfile.TarFile, tar_path: str
    ) -> Generator[ImageRecord, None, None]:
        members = list(self._image_members(tar_file))
        for member in tqdm(members, desc=f"Streaming {Path(tar_path).name}"):
            if self._at_limit():
                break
            file_obj = tar_file.extractfile(member)
            if file_obj is None:
                self._n_skipped += 1
                continue
            try:
                image = Image.open(file_obj)
                image.load()
                if self._config.convert_mode:
                    image = image.convert(self._config.convert_mode)
                if not self._check_size(image):
                    self._n_skipped += 1
                    continue
                self._n_yielded += 1
                yield self._to_record(member.name, image, tar_path)
            except Exception as exc:
                self._n_skipped += 1
                logger.warning("Failed to decode %s: %s", member.name, exc)
            finally:
                file_obj.close()

    def _stream_concurrent(
        self, tar_file: tarfile.TarFile, tar_path: str
    ) -> Generator[ImageRecord, None, None]:
        member_iter = self._image_members(tar_file)
        max_workers = self._concurrency.max_workers
        chunk_size = self._concurrency.chunk_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            while not self._at_limit():
                chunk_data = []
                for member in islice(member_iter, chunk_size):
                    if self._at_limit():
                        break
                    file_obj = tar_file.extractfile(member)
                    if file_obj is not None:
                        chunk_data.append((member.name, file_obj.read()))
                        file_obj.close()
                if not chunk_data:
                    break

                futures = {
                    pool.submit(self._decode, name, raw): name
                    for name, raw in chunk_data
                }
                for future in concurrent.futures.as_completed(futures):
                    name, image = future.result()
                    if image is not None:
                        self._n_yielded += 1
                        yield self._to_record(name, image, tar_path)
                    else:
                        self._n_skipped += 1

    def stream(self) -> Generator[ImageRecord, None, None]:
        self._n_yielded = 0
        self._n_skipped = 0

        for tar_path in self._config.tar_paths:
            if self._at_limit():
                break
            logger.info("Opening archive: %s", tar_path)
            with tarfile.open(tar_path, mode="r:*") as tar_file:
                if self._concurrency.enabled:
                    yield from self._stream_concurrent(tar_file, tar_path)
                else:
                    yield from self._stream_sequential(tar_file, tar_path)

        logger.info(
            "Stream complete: %d yielded, %d skipped",
            self._n_yielded,
            self._n_skipped,
        )

    def manifest(self) -> dict[str, list[str]]:
        result = {}
        for tar_path in self._config.tar_paths:
            with tarfile.open(tar_path, mode="r:*") as tf:
                result[tar_path] = [
                    m.name for m in tf.getmembers() if self._is_image(m)
                ]
        return result

    @property
    def stats(self) -> dict[str, int]:
        return {
            "yielded": self._n_yielded,
            "skipped": self._n_skipped,
            "total": self._n_yielded + self._n_skipped,
        }