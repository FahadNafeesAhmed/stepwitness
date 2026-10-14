"""Split each IndustReal state mesh into its loose parts and diff consecutive states. Runs inside Blender.

Each stateN.fbx is one merged mesh. The physical parts are separate shells inside it, so separating by
loose parts recovers them. Knowing which shells appear between two states tells us which shells make up
each component, which is what the render factory needs to build error states by moving or swapping parts.

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_split_states.py -- \
        --fbx-dir data/industreal/part_geometries/part_geometries --out runs/phase2/state_parts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def shell_signature(ob) -> dict:
    verts = len(ob.data.vertices)
    polys = len(ob.data.polygons)
    corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector(min(c[i] for c in corners) for i in range(3))
    hi = Vector(max(c[i] for c in corners) for i in range(3))
    center = (lo + hi) / 2
    dims = hi - lo
    mats = [m.name for m in ob.data.materials if m]
    return {
        "verts": verts,
        "polys": polys,
        "center": [round(v, 2) for v in center],
        "dims": [round(v, 2) for v in dims],
        "materials": mats,
    }


def split_state(fbx: Path) -> list[dict]:
    reset_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    shells = [shell_signature(o) for o in bpy.data.objects if o.type == "MESH"]
    shells.sort(key=lambda s: (s["center"][0], s["center"][1], s["center"][2]))
    return shells


def key(shell: dict) -> tuple:
    # a shell is identified by its size and position, rounded to the millimetre
    return (shell["verts"], tuple(round(v) for v in shell["center"]), tuple(round(v) for v in shell["dims"]))


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    files = sorted(args.fbx_dir.glob("state*.fbx"), key=lambda p: int(p.stem[5:]))
    per_state = {}
    for f in files:
        shells = split_state(f)
        per_state[f.stem] = shells
        print(f"{f.stem}: {len(shells)} shells; verts per shell: {sorted(s['verts'] for s in shells)}")

    # diff consecutive states in the assembly sequence 1..13 and the maintenance sequence 13..22
    diffs = {}
    order = [f"state{i}" for i in range(1, 23)]
    for a, b in zip(order, order[1:]):
        ka = {key(s) for s in per_state[a]}
        kb = {key(s) for s in per_state[b]}
        diffs[f"{a}->{b}"] = {
            "added": [list(map(list, (k[1], k[2]))) + [k[0]] for k in sorted(kb - ka)],
            "removed": [list(map(list, (k[1], k[2]))) + [k[0]] for k in sorted(ka - kb)],
        }
        print(f"{a}->{b}: +{len(kb - ka)} shells, -{len(ka - kb)} shells")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"states": per_state, "diffs": diffs}, indent=2))


if __name__ == "__main__":
    main()
