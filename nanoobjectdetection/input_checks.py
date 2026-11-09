"""Interactive checks for input image acquisition and localization settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .storage import iter_tiff_frames


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def check_saturation(
    input_file: Path,
    frame_start: int,
    frame_stop: int | None,
    y_slice: slice,
    x_slice: slice,
) -> float | None:
    """Plot brightest-frame intensity and ask whether its maximum is saturation."""
    maximum = None
    maximum_frame_count = 0
    maximum_frames: list[np.ndarray] = []

    for _, image in iter_tiff_frames(input_file, frame_start, frame_stop):
        image_roi = image[y_slice, x_slice]
        frame_maximum = np.max(image_roi)
        if maximum is None or frame_maximum > maximum:
            maximum = frame_maximum
            maximum_frame_count = 1
            maximum_frames = [image_roi.copy()]
        elif frame_maximum == maximum:
            maximum_frame_count += 1
            if len(maximum_frames) < 10:
                maximum_frames.append(image_roi.copy())

    if maximum is None:
        raise ValueError("No image frames are available for the saturation check")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.hist(np.concatenate(maximum_frames).ravel(), bins=2**8, log=True)
    axis.set_title("Intensity histogram of frames containing the maximum pixel")
    axis.set_xlabel("Intensity")
    axis.set_ylabel("Counts")
    plt.show(block=True)

    try:
        is_saturated = ask_yes_no(
            "Does the highest-intensity histogram bin indicate image saturation?"
        )
    finally:
        plt.close(figure)

    if is_saturated:
        print(
            f"Using the observed maximum ({float(maximum):g}) as the saturation threshold "
            f"(present in {maximum_frame_count} frame(s))."
        )
        return float(maximum)

    print("Treating the image stack as unsaturated.")
    return None
