# ORB SLAM Implementation

This repository contains a Pythonic implementation of the ORB SLAM algorithm based on the paper:
"ORBSLAM-based Endoscope Tracking and 3D Reconstruction" by Nader Mahmoud, et al.

## Introduction

This implementation includes the tracking and mapping components of the ORB SLAM algorithm. The system tracks the endoscope location in a live video, estimates 3D map points, and performs 3D reconstruction using Bundle Adjustment.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- g2o (for Bundle Adjustment)
## Reference Paper

Please refer to the original paper for detailed information on the ORB SLAM algorithm:
[ORBSLAM-based Endoscope Tracking and 3D Reconstruction](https://arxiv.org/abs/1608.08149)

## Work in Progress
This implementation is a work in progress. Feel free to contribute if you've read the paper or have some suggestions.

Install dependencies using the following command:

```bash
pip install opencv-python numpy g2o



