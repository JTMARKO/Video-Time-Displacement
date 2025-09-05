import os
from typing import Optional, Sequence, Union

import cv2
# import cupy as np
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm


class Modifier:
    """Essentially represents a 3d numpy array. It should be filled with 3d perlin noise. All values should be between -1 and 1"""

    def __init__(
        self,
        width,
        height,
        length,
        density: int = 300,
    ) -> None:
        self.width = width
        self.height = height
        self.length = length
        self.density = density

        self.z_range = np.linspace(0, 1, length)

        self.modifiers = []

    def _generate_modifier_slice(self, z) -> np.ndarray: ...

    def add_modifier(self, modifier: np.ndarray) -> None:
        """Adds a 2d array that is multiplied across the entire length, example use case is to create a mask"""
        if modifier.shape != (self.height, self.width):
            raise ValueError(
                "Modifier shape must match the height and width of the noise: {}".format(
                    (self.height, self.width)
                )
            )
        self.modifiers.append(modifier)

    def modifier_iterate(self):
        # for z in range(self.length):
        for z in range(self.length):
            slice = self._generate_modifier_slice(z)

            for m in self.modifiers:
                slice *= m
            yield slice

    def view_modifier(self):
        frame_width = self.width
        frame_height = self.height

        cwd = os.getcwd()
        file_name = str(self) + "_modifier.mp4"
        # file_name = os.path.join("Test", file_name)
        file_name = os.path.join(cwd, file_name)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Use 'mp4v' for mp4
        out = cv2.VideoWriter(file_name, fourcc, 30, (frame_width, frame_height))

        for z in tqdm(self.z_range):
            slice_noise = self._generate_modifier_slice(z * self.length)
            slice_noise = (slice_noise - slice_noise.min()) / (
                slice_noise.max() - slice_noise.min()
            )
            slice_noise = (slice_noise * 255).astype(np.uint8)
            for m in self.modifiers:
                slice_noise *= m
            # yield slice_noise
            out.write(cv2.cvtColor(slice_noise, cv2.COLOR_GRAY2BGR))

        out.release()
        cv2.destroyAllWindows()
        print("Saved video: " + file_name)
        return

    def __str__(self) -> str:
        name = f"{self.width}_{self.height}_{self.length}_{self.density}"
        return name


class WorleyNoiseModifier(Modifier):

    def __init__(
        self,
        width,
        height,
        length,
        density: int = 300,
        z_range=None,
    ) -> None:
        # self.points = points
        super().__init__(width, height, length, density)

        self.z_range = np.linspace(0, 1, length) if z_range is None else z_range
        points = np.random.randint(
            0,
            [length * max(self.z_range), height, width],
            size=(density, 3),
        )

        self.tree = cKDTree(points)

    def _generate_modifier_slice(self, z):
        width, height = self.width, self.height
        # Create a grid of points at the given z slice
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        coords = np.stack([np.full_like(xx, z), yy, xx], axis=-1).reshape(-1, 3)
        distances, _ = self.tree.query(coords, k=1)
        slice_noise = distances.reshape((height, width)).astype(np.float16)
        # Normalize noise to range [0,1]
        return slice_noise

    def modifier_iterate(self):
        # for z in range(self.length):
        for z in self.z_range:
            slice_noise = self._generate_modifier_slice(z * self.length)
            slice_noise = (slice_noise - slice_noise.min()) / (
                slice_noise.max() - slice_noise.min()
            )
            for m in self.modifiers:
                slice_noise *= m
            yield slice_noise


class FileModifier(Modifier):
    def __init__(
        self, width, height, length, file_name: str, density: int = 300
    ) -> None:
        # self.frame =
        image = Image.open(file_name)
        self.frame = np.array(image)[:, :, 0] / 255.0

        if (height, width) != self.frame.shape:
            raise ValueError(
                f"Modifier image must be of video shape {height}x{width}",
                "Intead got",
                self.frame.shape,
            )

        super().__init__(width, height, length, density)

    def _generate_modifier_slice(self, z) -> np.ndarray:
        return self.frame.copy()


def modifier_factory(
    width, height, length, density, z_range, file_name: Optional[str]
) -> Modifier:

    if file_name:
        return FileModifier(width, height, length, file_name, density)
    else:
        return WorleyNoiseModifier(width, height, length, density, z_range)
