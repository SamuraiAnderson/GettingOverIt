# NanoTrack v2 local inference weights

These two ONNX files are used by the offline pot tracker. No training or package installation is performed.

The weights are excluded from Git. Before rerunning extraction in a fresh checkout, download the two files below from the upstream directory into this folder and verify their SHA-256 hashes. PPO corridor training only needs the extracted trajectory JSON.

Official API and model reference: https://docs.opencv.org/4.x/d8/d69/classcv_1_1TrackerNano.html

Upstream model directory: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

- `nanotrack_backbone_sim.onnx`: SHA-256 `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4`
- `nanotrack_head_sim.onnx`: SHA-256 `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9`

Downloaded from the corresponding `raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2/` paths. See upstream for model terms and attribution. Runtime output records the file hashes and OpenCV version in `rigid_tracker_report.json`.
