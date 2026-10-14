"""Render one small image per loose shell of a state mesh with that shell highlighted in green, then
compose them into a contact sheet. Used once to decide, by eye, which shell is which physical part.
Runs inside Blender; composition of the sheet uses Pillow afterwards (tools/compose_sheet.py).

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_highlight_shells.py -- \
        --fbx data/industreal/part_geometries/part_geometries/state13.fbx --out runs/phase2/shells_state13
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[480, 300])
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = args.size
    scene.render.image_settings.file_format = "JPEG"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    bpy.ops.import_scene.fbx(filepath=str(args.fbx))
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

    # same ordering as blender_split_states.py: by centre
    def centre(ob):
        cs = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        lo = Vector(min(c[i] for c in cs) for i in range(3))
        hi = Vector(max(c[i] for c in cs) for i in range(3))
        return (lo + hi) / 2, hi - lo

    shells.sort(key=lambda o: tuple(round(v, 2) for v in centre(o)[0]))

    # scene bounds, camera above looking down at 55 degrees, two lights
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in shells:
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector(min(a, b) for a, b in zip(lo, w))
            hi = Vector(max(a, b) for a, b in zip(hi, w))
    center = (lo + hi) / 2
    radius = (hi - lo).length / 2
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(50)
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    el = math.radians(60)
    cam.location = center + Vector((0, -radius * 2.6 * math.cos(el), radius * 2.6 * math.sin(el)))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    for i, pos in enumerate([(1, -1, 2), (-1, 1, 2)]):
        ld = bpy.data.lights.new(f"L{i}", "SUN")
        ld.energy = 2.0
        lo_ = bpy.data.objects.new(f"L{i}", ld)
        bpy.context.collection.objects.link(lo_)
        lo_.location = center + Vector(pos) * radius * 3
        lo_.rotation_euler = (center - lo_.location).to_track_quat("-Z", "Y").to_euler()

    green = bpy.data.materials.new("Highlight")
    green.use_nodes = True
    bsdf = green.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 1.0, 0.1, 1.0)
    bsdf.inputs["Emission Color"].default_value = (0.05, 1.0, 0.1, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.5

    index = []
    for i, ob in enumerate(shells):
        saved = [ob.data.materials[j] for j in range(len(ob.data.materials))]
        for j in range(len(ob.data.materials)):
            ob.data.materials[j] = green
        scene.render.filepath = str(out / f"shell_{i:02d}.jpg")
        bpy.ops.render.render(write_still=True)
        for j, m in enumerate(saved):
            ob.data.materials[j] = m
        c, d = centre(ob)
        index.append({"index": i, "center": [round(v, 2) for v in c], "dims": [round(v, 2) for v in d], "verts": len(ob.data.vertices)})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"rendered {len(shells)} shell highlights to {out}")


if __name__ == "__main__":
    main()
