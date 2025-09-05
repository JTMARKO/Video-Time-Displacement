# Video Time Displacement

This script compiles a video with noise modifiers based on an input video and a modifier file. It uses the red channel of the modifier file to determine the displacement of frames in the output video.

## Installation

1.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```
2.  **Activate the virtual environment:**

    *   On Linux/macOS:

        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:

        ```bash
        .venv\Scripts\activate
        ```
3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```
python cli.py --input_file <input_video_path> --file_name <output_video_name> --fps <frames_per_second> --seconds_ahead <seconds_ahead> --modifier_file_name <modifier_file_path>
```

## Arguments

*   `--input_file`: Path to the input video file. (Required)
*   `--file_name`: Name of the output video file. (Required)
*   `--fps`: Frames per second for the output video. (Default: 60)
*   `--seconds_ahead`: Seconds ahead for vfxvideo. (can be a negative number) (Default: 1)
*   `--modifier_file_name`: Path to the modifier file. Looks at the red channel. R=0 => no change, R=255 => maximum displacement. (Required)

## Example

```
python cli.py --input_file input.mp4 --file_name output.mp4 --fps 30 --seconds_ahead 2 --modifier_file_name modifier.png
```

This command will compile `input.mp4` with noise modifiers, using `modifier.png` as the modifier file, and save the output as `output.mp4` with 30 frames per second, with the video displacement being 2 seconds ahead.

## To-Do

* Add support for displacement video. This will make all functionality of just running files and editing them available to the command line
