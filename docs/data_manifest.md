# Data manifest

Nothing under `data/` is committed. This file records where each file came from and its checksum,
so the data folder can be rebuilt exactly.

Source: IndustReal dataset, 4TU.ResearchData, version 2 (2024-08-23).
DOI 10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224
Base URL: `https://data.4tu.nl/file/b008dd74-020d-4ea4-a8ba-7bb60769d224/<file-id>`

| File | Bytes | MD5 | File id | Used for |
|---|---:|---|---|---|
| part_geometries.zip | 17,338,432 | 20a25a8eaa2ad605f269d30fa40275a7 | 4b6641cf-0034-4875-8ebb-73e35e4a5b91 | render factory (phase 2) |
| ASD_results_IndustRealplusSynthetic_test.zip | 733,682 | fd37671516f15d52c8bf8d77964b4b18 | 2b348dfc-cecd-4856-9de8-6d948d2e5547 | baseline predictions, real-trained (phase 0) |
| ASD_results_SyntheticOnly_test.zip | 1,147,376 | 9c24b416143df4ed4eb97464023d02d9 | f659cc29-c81d-43e9-a94d-27255f33fd33 | baseline predictions, synthetic-only (phase 0) |
| action_recognition_labels.zip | 67,816 | 3a9c9e5b0060065edcadb0375b26e0ee | 778d7d5f-6631-4cb6-9451-f88c574c7991 | action labels, reference only |
| all_rgb_videos.zip | 4,960,644,152 | c5b8901dba179d2eb10e9348f8163c79 | cf21602e-4424-42a4-b05d-9897957082ad | every recording as mp4 (runtime input) |
| test_p1.zip | 7,535,688,635 | 04f53405ca41d85681ef9323c588d271 | ec7a4f94-7c1d-497d-b75a-4e694b7a9f7e | test labels and frames |
| test_p2.zip | 5,952,711,048 | c76b3ca8a03f2c67764cb9c16eff358d | d07789b6-83b1-429d-be0c-95edba210757 | test labels and frames |
| test_p3.zip | 10,236,464,186 | 7a81827d26293520d3da87ff77c6abd4 | 31f88cb6-a2e7-44df-ac51-5aaa0a148fc3 | test labels and frames |

Not downloaded: train and validation recordings (26 GB), the authors' synthetic set (16 GB), the
action-recognition and state-detection model weights. The project trains on its own renders.

Reference code: `third_party/IndustReal`, cloned from https://github.com/TimSchoonbeek/IndustReal
(Apache-2.0), commit ad86a35.

Download script for the large files: `data/download_big.sh` (resumable, verifies MD5, logs to
`data/download_big.log`).
