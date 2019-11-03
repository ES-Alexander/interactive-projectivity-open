# interactive-projectivity-open
A framework for detecting interactions with projections using computer vision on a camera feed.

## About
Developed as part of my honours thesis as a modular, expandable framework allowing for simple tracking and control of interactions with a projector screen. Potential uses include mirroring a laser pointer onto multiple screens during a presentation, and/or for a recording, and with sufficient algorithmic advances the possibility of remotely controlling a computer via a large interactive projection, using one or more laser pointers, fingers, or other interaction methods.

## Features
- Robust screen detection
- Background lighting compensation
- Background subtraction using screenshots of screen being projected
- Real-time detection and tracking of a laser-pointer on a screen
- Mouse control for following detected interactions
- Kalman compensation for improved tracking of interactions

## Installation and Running
Clone or download the repository code to your computer.
Install `pyautogui`, `mss`, `numpy`, `opencv-python`, `overrides`, and `matplotlib` using `pip`.

Navigate to where you have the repository code, and run `controller.py` from the command line (use `python3 controller.py --help` for help and options).

## Attributions
This project builds upon and uses existing open-source projects as below:
- a significantly modified form of the Kalman Filter code provided at https://github.com/Myzhar/simple-opencv-kalman-tracker, available under the LGPLv3
- `pyautogui` for mouse control, available under BSD 3-Clause "New" or "Revised" License
- `mss` for fast cross-platform screenshots, available under the MIT License
- `opencv-python` for computer vision, available under 3-Clause BSD License
- `overrides` for checking overrides of abstract methods, available under Apache License, Version 2.0
- `numpy` for fast array processing, available under BSD License
