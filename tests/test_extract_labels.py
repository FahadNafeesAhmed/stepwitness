import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_zip(path: Path, split: str, rec: str, n_frames: int) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        base = f"recordings/{split}/{rec}"
        zf.writestr(f"{base}/PSR_labels.csv", "000100.jpg,3,Install front chassis\n")
        zf.writestr(f"{base}/PSR_labels_raw.csv", "000000.jpg,1,0,0,0,0,0,0,0,0,0,0\n")
        zf.writestr(f"{base}/OD_labels.json", json.dumps({"x": 1}))
        zf.writestr(f"{base}/hands.csv", "a,b\n")
        for i in range(n_frames):
            zf.writestr(f"{base}/rgb/{i:06d}.jpg", b"\xff\xd8\xff")
        for i in range(3):
            zf.writestr(f"{base}/depth/{i:06d}.jpg", b"\xff\xd8\xff")
        zf.writestr(f"{base}/stereo_left/000000.jpg", b"\xff\xd8\xff")


def test_extracts_only_labels_and_counts_rgb_frames(tmp_path: Path):
    z1 = tmp_path / "test_p1.zip"
    z2 = tmp_path / "test_p2.zip"
    make_zip(z1, "test", "03_assy_0_1", 7)
    make_zip(z2, "test", "08_main_0_1", 4)
    out = tmp_path / "recordings"
    counts = tmp_path / "frame_counts.json"
    cmd = [sys.executable, str(ROOT / "tools" / "extract_labels.py"), str(z1), str(z2), "--out", str(out), "--counts", str(counts)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert (out / "test" / "03_assy_0_1" / "PSR_labels.csv").read_text().startswith("000100.jpg")
    assert (out / "test" / "03_assy_0_1" / "PSR_labels_raw.csv").exists()
    assert (out / "test" / "03_assy_0_1" / "OD_labels.json").exists()
    assert (out / "test" / "03_assy_0_1" / "hands.csv").exists()
    assert not (out / "test" / "03_assy_0_1" / "rgb").exists(), "frames must never be extracted"
    assert json.loads(counts.read_text()) == {"test/03_assy_0_1": 7, "test/08_main_0_1": 4}
