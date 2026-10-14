"""Inventory the IndustReal state FBX files: objects, mesh sizes, bounding boxes. Runs inside Blender.

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_inspect_fbx.py -- \
        --fbx-dir data/industreal/part_geometries/part_geometries --out runs/phase2/fbx_inventory.json

Everything after ``--`` is for this script. Blender's own arguments come before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy  # type: ignore  # available inside Blender only
from mathutils import Vector  # type: ignore


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def inventory_one(fbx: Path) -> dict:
    reset_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    objects = []
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for ob in bpy.data.objects:
        entry = {
            "name": ob.name,
            "type": ob.type,
            "parent": ob.parent.name if ob.parent else None,
            "location": [round(v, 4) for v in ob.matrix_world.translation],
            "dimensions": [round(v, 4) for v in ob.dimensions],
        }
        if ob.type == "MESH":
            entry["vertices"] = len(ob.data.vertices)
            entry["polygons"] = len(ob.data.polygons)
            entry["materials"] = [m.name for m in ob.data.materials if m]
            for corner in ob.bound_box:
                w = ob.matrix_world @ Vector(corner)
                lo = Vector(min(a, b) for a, b in zip(lo, w))
                hi = Vector(max(a, b) for a, b in zip(hi, w))
        objects.append(entry)
    meshes = [o for o in objects if o["type"] == "MESH"]
    return {
        "file": fbx.name,
        "n_objects": len(objects),
        "n_meshes": len(meshes),
        "scene_unit_scale": bpy.context.scene.unit_settings.scale_length,
        "bbox_min": [round(v, 4) for v in lo] if meshes else None,
        "bbox_max": [round(v, 4) for v in hi] if meshes else None,
        "objects": objects,
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    files = sorted(args.fbx_dir.glob("state*.fbx"), key=lambda p: int(p.stem[5:]))
    result = [inventory_one(f) for f in files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for r in result:
        names = ", ".join(o["name"] for o in r["objects"] if o["type"] == "MESH")
        print(f"{r['file']}: {r['n_meshes']} meshes, bbox {r['bbox_min']} .. {r['bbox_max']} :: {names}")


if __name__ == "__main__":
    main()
