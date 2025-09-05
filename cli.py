#! python
import argparse

from vfx_video import VideoCompiler


def parse_args():
    parser = argparse.ArgumentParser(description="Compile video with noise modifiers.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--file_name", type=str, required=True, help="Name of the output video file."
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Frames per second for the output video."
    )
    parser.add_argument(
        "--density", type=int, default=300, help="Density of the noise modifier."
    )
    parser.add_argument(
        "--seconds_ahead",
        type=int,
        default=1,
        help="seconds ahead for vfxvideo. (can be a negative number)",
    )
    parser.add_argument(
        "--modifier_file_name",
        type=str,
        required=True,
        help="Mandatory modifier file name. Looks at the red channel. R=0 => no change, R=255 => maximum displacement",
    )
    # parser.add_argument(
    #     "--bpm", type=int, required=True, help="Beats per minute for the wave."
    # )
    # parser.add_argument(
    #     "--wave_mime_type",
    #     type=str,
    #     required=True,
    #     help="Wave mime type (e.g., square).",
    # )
    # parser.add_argument("--scale", type=float, default=1.0, help="Scale for the wave.")

    args = parser.parse_args()

    compiler = VideoCompiler(
        input_file=args.input_file,
        file_name=args.file_name,
        fps=args.fps,
        density=args.density,
        seconds_ahead=args.seconds_ahead,
        modifier_file_name=args.modifier_file_name,
    )

    compiler.run()


if __name__ == "__main__":
    parse_args()
