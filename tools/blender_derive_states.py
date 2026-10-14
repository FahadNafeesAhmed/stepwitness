"""Validate the shell-to-component mapping: rebuild every labelled state from a complete model by
deleting component groups, and render it next to the dataset's own file for that state.
Runs inside Blender.

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_derive_states.py -- \
        --fbx-dir data/industreal/part_geometries/part_geometries --out runs/phase2/derive
Then: python tools/compose_pairs.py runs/phase2/derive --out runs/phase2/derive/sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import labelled_states_industreal, state_str  # noqa: E402

MAPPING = json.loads((ROOT / "stepwitness" / "procedures" / "industreal_shells.json").read_text())


def setup_render(size=(480, 300)) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.image_settings.file_format = "JPEG"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0


def import_and_split(fbx: Path) -> list:
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    for ob in list(bpy.data.objects):
        if ob.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(ob, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    shells = [o for o in bpy.data.objects if o.type == "MESH"]

    def centre(ob):
        cs = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        lo = Vector(min(c[i] for c in cs) for i in range(3))
        hi = Vector(max(c[i] for c in cs) for i in range(3))
        return (lo + hi) / 2

    shells.sort(key=lambda o: tuple(round(v, 2) for v in centre(o)))
    return shells


def bounds(objects):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in objects:
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector(min(a, b) for a, b in zip(lo, w))
            hi = Vector(max(a, b) for a, b in zip(hi, w))
    return lo, hi


def place_camera_and_lights(center: Vector, radius: float) -> None:
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(50)
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    el = math.radians(55)
    cam.location = center + Vector((radius * 0.6, -radius * 2.4 * math.cos(el), radius * 2.4 * math.sin(el)))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    for i, pos in enumerate([(1, -1, 2), (-1, 1, 2)]):
        ld = bpy.data.lights.new(f"L{i}", "SUN")
        ld.energy = 2.0
        lo_ = bpy.data.objects.new(f"L{i}", ld)
        bpy.context.collection.objects.link(lo_)
        lo_.location = center + Vector(pos) * radius * 3
        lo_.rotation_euler = (center - lo_.location).to_track_quat("-Z", "Y").to_euler()


def render_to(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)


def derive(target: tuple, fbx_dir: Path, out: Path, name: str) -> None:
    source_key = "state22" if target[4] == 1 else "state13"
    spec = MAPPING[source_key]
    setup_render()
    shells = import_and_split(fbx_dir / spec["source"])
    assert len(shells) == 35, f"{spec['source']}: expected 35 shells, got {len(shells)}"
    full_lo, full_hi = bounds(shells)
    center = (full_lo + full_hi) / 2
    radius = (full_hi - full_lo).length / 2
    for comp, idx in spec["components"].items():
        if target[int(comp)] != 1:
            for i in idx:
                bpy.data.objects.remove(shells[i], do_unlink=True)
    place_camera_and_lights(center, radius)
    render_to(out / f"{name}_derived.jpg")


def original(state_no: int, fbx_dir: Path, out: Path, name: str) -> None:
    setup_render()
    shells = import_and_split(fbx_dir / f"state{state_no}.fbx")
    lo, hi = bounds(shells)
    # frame the original the way the full model is framed: same radius as the complete car
    ref = MAPPING["state22" if state_no >= 19 else "state13"]["source"]
    radius = 95.0  # half-diagonal of the complete car, so scale matches the derived render
    place_camera_and_lights((lo + hi) / 2, radius)
    render_to(out / f"{name}_original.jpg")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", default="all")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    labelled = labelled_states_industreal()
    wanted = list(range(1, 23)) if args.states == "all" else [int(s) for s in args.states.split(",")]
    for n in wanted:
        target = labelled[n]
        name = f"state{n:02d}_{state_str(target)}"
        derive(target, args.fbx_dir, args.out, name)
        original(n, args.fbx_dir, args.out, name)
        print(f"state{n}: derived and original rendered")


if __name__ == "__main__":
    main()
