# Deploy

On-board ROS 2 / C++ runtime for the G1. Mirrors `unitree_rl_lab/deploy/` layout.

```
deploy/
├── robots/
│   └── g1_23dof/           # C++ controller (mirrors unitree_rl_lab/deploy/robots/g1_29dof)
├── include/                # FSM, isaaclab-side bindings
├── pointcloud_to_bev.py    # On-board BEV builder (10 Hz)
├── g1_runtime.py           # 50 Hz policy node
└── safety_monitor.py
```

See §14–15 of `../STAIR_CLIMBING_REPLICATION_GUIDE.md`.
