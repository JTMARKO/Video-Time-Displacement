from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# from scipy.spatial import cKDTree
# from dilation import Modifier, WorleyNoiseModifier
from dilation import Modifier, modifier_factory
from masks import *
from util import fractional_repeat


class VfxVideo:

    def __init__(
        self, file_path: str, frame_range: int, cached_name: Optional[str] = None
    ) -> None:
        self.frame_range = frame_range
        self.file_path = file_path
        self.cached_name = cached_name
        self.video = self._load_video()

    def _load_video(self) -> np.ndarray:
        """Loads the video from the file path and converts it to a numpy array."""
        cap = cv2.VideoCapture(self.file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        return np.array(frames)

    def add_noise_modifier(self, noise_modifier: Modifier):
        self.noise_modifier = noise_modifier

    def combine(self):
        # cached_name = self.cached_name
        # if cached_name:
        #     modified_noise = np.load(cached_name) * self.frame_range
        # else:
        #     modified_noise_basic_range = self.noise_modifier.get_modified_noise()
        #     np.save(str(self.noise_modifier), modified_noise_basic_range)
        #     modified_noise = modified_noise_basic_range * self.frame_range

        video_frames = self.video
        num_frames, height, width, channels = video_frames.shape
        num_frames = min(num_frames, self.noise_modifier.length)

        # Prepare output array
        new_video = np.empty_like(video_frames)

        # Integer noise values
        # noise_indices = modified_noise[:num_frames].astype(int)

        # For each frame, compute the source frame indices clamped within range
        for frame_idx, noise_slice in zip(
            tqdm(range(num_frames)),
            self.noise_modifier.modifier_iterate(),
        ):

            slice = (noise_slice * self.frame_range).astype(int)
            # Create an array of source frame indices for the entire frame
            source_frame_idx = np.clip(
                # frame_idx + noise_indices[frame_idx], 0, num_frames - 1
                frame_idx + slice,
                0,
                num_frames - 1,
            )

            # Use advanced indexing to select pixels from their source frames
            new_video[frame_idx] = video_frames[
                source_frame_idx, np.arange(height)[:, None], np.arange(width)
            ]

        return new_video


class FileVideoPermanence:

    def __init__(self, video: VfxVideo, file_name: str) -> None:
        self.file_name = file_name
        self.video = video

    def save(
        self,
        fps=30,
    ) -> None:
        # return super().save(start_frame, end_frame)
        vfx_video = self.video

        video = vfx_video.combine()

        # fps = fps
        frame_width = int(vfx_video.video[0].shape[1])
        frame_height = int(vfx_video.video[0].shape[0])

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Use 'mp4v' for mp4
        out = cv2.VideoWriter(self.file_name, fourcc, fps, (frame_width, frame_height))

        # Write the frames to the output video
        for i in range(video.shape[0]):
            out.write(video[i].astype(np.uint8))

        # Release everything
        out.release()
        cv2.destroyAllWindows()
        print("Saved video: " + self.file_name)
        return


class VideoCompiler:
    def __init__(
        self,
        input_file: str,
        file_name: str,
        fps: int = 30,
        density=300,
        seconds_ahead=1,
        modifier_file_name: Optional[str] = None,
    ) -> None:
        # self.input_file = input_file
        self.file_name = file_name
        self.fps = fps
        self.density = density
        self.modifier_file_name = modifier_file_name

        video = VfxVideo(input_file, fps * seconds_ahead)
        self.length, _, _, _ = video.video.shape
        self.video = video

        self.z_range = None
        return

    def run(self, mask=None):
        # video = VfxVideo(self.input_file, self.fps)
        video = self.video
        length, height, width, _ = video.video.shape
        noise_modifier = modifier_factory(
            width, height, length, self.density, self.z_range, self.modifier_file_name
        )
        # noise_modifier.view_modifier()
        # return
        # print(self.z_range)
        # for frame in noise_modifier.blob_noise_iterate():
        #     plt.imshow(frame)
        #     plt.show()
        # return

        if mask is not None:
            noise_modifier.add_modifier(mask)

        video.add_noise_modifier(noise_modifier)

        permanence = FileVideoPermanence(video, self.file_name)
        permanence.save(self.fps)
        pass

    def view_frame(self, frame_no=0):
        video = self.video
        frame = video.video[frame_no]
        plt.imshow(frame)
        plt.show()

    def create_z_range(self, bpm, wave_mime_type, scale: float = 1):
        bps = bpm / 60
        length_time = self.length / self.fps

        total_beats = int(bps * length_time)

        if wave_mime_type == "square":
            dense_wave = np.linspace(0, 1, total_beats)
            repeat = self.length / total_beats
            wave = fractional_repeat(dense_wave, repeat, self.length)
            # np.repeat(
            #     dense_wave,
            # )
        # elif wave_mime_type == "sawtooth":
        #     pass
        #
        # elif wave_mime_type == "sin":
        #     pass

        else:
            raise ValueError(f"wave_mime_type {wave_mime_type} not found")

        self.z_range = wave * scale
        # print(self.z_wave * self.length)
        return


def main():
    mask = create_bridge_blurred_mask()
    # plt.imshow(mask)
    # plt.show()
    # return
    music_video = VideoCompiler(
        "./Data/bridge_middle.AVI",
        "./TESTER.mp4",
        density=400,
        seconds_ahead=-1,
        # modifier_file_name="./example_frame.png",
    )
    # music_video.view_frame(50)
    # return
    # return
    # music_video.create_z_range(97.5, "square", 2)
    # length = music_video.length
    music_video.run(mask)

    # music_video.run(create_blurred_mask())
    # music_video.run(create_bridge_blurred_mask())
    return

    width = 640
    height = 480
    length = 1200
    density = 300
    # frame_range = 10
    # file_name = f"{width}_{height}_{length}_{density}.npy"

    # noise_modifier = NoiseModifier(width, height, length, frame_range)
    z_range = np.linspace(0, 0.1, length)
    z_range = None
    noise_modifier = WorleyNoiseModifier(width, height, length, density, z_range)

    cached_file_name = str(noise_modifier)
    video = VfxVideo("./Data/cars.AVI", 30)
    video.add_noise_modifier(noise_modifier)

    permanence = FileVideoPermanence(video, "./Build/cars_morph_many.mp4")
    permanence.save()
    return

    noise = noise_modifier.get_modified_noise()
    # Display a slice of the 3D noise as an image
    image_slice = noise[50, :, :]  # Take a slice at the middle of the length
    # Scale the noise values to be between 0 and 255
    image_slice = ((image_slice + 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_slice, mode="L")  # "L" mode for grayscale
    img.show()

    image_slice = noise[99, :, :]  # Take a slice at the middle of the length
    # Scale the noise values to be between 0 and 255
    image_slice = ((image_slice + 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_slice, mode="L")  # "L" mode for grayscale
    img.show()


if __name__ == "__main__":
    main()
