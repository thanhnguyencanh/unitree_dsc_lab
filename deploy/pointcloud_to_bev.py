"""On-board point-cloud -> 6-channel BEV (mirrors training-time bev.py).

Subscribes to RealSense /camera/depth/color/points (or Livox PointCloud2) and
publishes the BEV tensor consumed by encoder.onnx.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("ROS 2 node: depth subscriber -> BEV publisher.")


if __name__ == "__main__":
    main()
