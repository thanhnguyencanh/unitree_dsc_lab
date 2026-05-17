"""On-board 50 Hz policy node (paper §IV, Fig. 2).

  RealSense depth (30 Hz) -> pointcloud_to_bev -> encoder.onnx (10 Hz) -> z_t
                                                                          \\
  unitree state (500 Hz) -> proprio buffer ----------------------------- policy.onnx (50 Hz)
                                                                          |
                                                                 joint targets -> LowCmd
"""

from __future__ import annotations


def main():
    raise NotImplementedError(
        "Wire onnxruntime sessions for encoder/policy + unitree_sdk2py LowCmd publishing."
    )


if __name__ == "__main__":
    main()
