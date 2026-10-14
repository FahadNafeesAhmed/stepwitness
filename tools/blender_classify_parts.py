"""Classify the loose shells of any IndustReal state file into part classes by geometry, and assign
each to a procedure component. Runs inside Blender; the same functions are imported by the renderer
to build error states.

Part classes (sizes in mm, the files are at the dataset's 200 percent print scale):
  wheel         black, about 37 x 13 x 37
  wheel_nut     magenta disc near a wheel
  axle          silver, 60 to 85 mm long, thin
  pin           silver, about 32 mm long, 11 mm thick (chassis pins)
  pin_clip      magenta disc within 20 mm of a pin's axis
  knob          magenta, taller than 18 mm, about 11 mm wide (thumb screw)
  brace_long    white, about 75 x 3.5 x 25 (rear chassis, long braces)
  brace_short   white, about 50 x 3.3 x 12 (short rear chassis)
  plate         white, about 50 x 12 x 3 (front chassis plates)
  block         white or magenta, about 25 x 37 x 12 or 50 x 25 x 12 (base, bracket)
  beam          white, about 50 x 13 x 13
  pulley        magenta, about 16 x 12 x 16 at the rear axle
  other         anything else

Component assignment uses the car's long axis: the rear is the end where braces are, the front is
the opposite end. Rear pins are the horizontal pins near the braces, ordered along the axis so the
one nearer the rear end is the rear-rear pin. The vertical pin on the plates is the front chassis
pin. Wheels and their nuts go to the rear or front wheel assembly by which end they are nearer.

Usage (sheet for a visual check):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_classify_parts.py -- \
        --fbx data/industreal/part_geometries/part_geometries/state13.fbx --out runs/phase2/classify_state13
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

CLASS_COLOURS = {
    "wheel": (0.1, 0.1, 0.1),
    "wheel_nut": (1.0, 0.5, 0.0),
    "axle": (0.0, 0.6, 1.0),
    "pin": (0.0, 1.0, 0.2),
    "pin_clip": (0.6, 1.0, 0.0),
    "knob": (1.0, 0.0, 1.0),
    "brace_long": (1.0, 1.0, 0.0),
    "brace_short": (1.0, 0.8, 0.4),
    "plate": (0.0, 1.0, 1.0),
    "block": (0.5, 0.0, 1.0),
    "beam": (0.3, 0.3, 1.0),
    "pulley": (1.0, 0.3, 0.3),
    "other": (1.0, 1.0, 1.0),
}


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------

def split_shells(meshes: list) -> list:
    bpy.ops.object.select_all(action="DESELECT")
    for ob in meshes:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    shells = [o for o in bpy.data.objects if o.type == "MESH"]
    for ob in shells:
        ob["is_assembly"] = True
    return shells


def world_bounds(ob):
    cs = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector(min(c[i] for c in cs) for i in range(3))
    hi = Vector(max(c[i] for c in cs) for i in range(3))
    return lo, hi


def dominant_colour(ob) -> str:
    counts: dict[str, int] = {}
    for poly in ob.data.polygons:
        slot = ob.data.materials[poly.material_index] if poly.material_index < len(ob.data.materials) else None
        family = slot.name.split(".")[0] if slot else "none"
        counts[family] = counts.get(family, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def near(v: float, target: float, tol: float) -> bool:
    return abs(v - target) <= tol


def classify_shell(ob) -> dict:
    lo, hi = world_bounds(ob)
    d = hi - lo
    c = (lo + hi) / 2
    colour = dominant_colour(ob)
    dims = sorted([d.x, d.y, d.z], reverse=True)  # longest first
    L, M, S = dims
    cls = "other"
    if colour == "Black" and L > 30:
        cls = "wheel"
    elif colour == "Metallic silver":
        if L >= 55 and M <= 14:
            cls = "axle"
        elif near(L, 32, 5) and M <= 13:
            cls = "pin"
        elif L < 30:
            cls = "other"  # small screws and nuts
    elif colour == "Magenta":
        if L >= 18 and M <= 16 and S >= 8:
            cls = "knob"  # the thumb knob of the front chassis pin: tall and narrow
        elif near(L, 16, 2) and near(M, 16, 3) and S >= 10:
            cls = "pulley"
        elif L <= 14 and S <= 6:
            cls = "disc"  # resolved below into wheel_nut or pin_clip
        elif L >= 24 and M >= 24:
            cls = "block"
    elif colour == "White":
        if L >= 70 and S <= 5:
            cls = "brace_long"
        elif near(L, 50, 4) and S <= 5 and M <= 15:
            # plates and short braces share 50 x 12 x 3; plates lie flat (thin axis z), braces stand (thin axis y)
            cls = "plate" if d.z <= 5 else "brace_short"
        elif L >= 24 and M >= 24 and S >= 10:
            cls = "block"
        elif near(L, 50, 4) and M >= 11 and S >= 11:
            cls = "beam"
    return {"name": ob.name, "class": cls, "colour": colour, "center": list(c), "dims": [d.x, d.y, d.z], "lo": list(lo), "hi": list(hi)}


def classify_file(shells: list) -> list[dict]:
    parts = [classify_shell(ob) for ob in shells]
    # the car's long axis is x in these files; the rear is where the braces (long or short) sit
    braces = [p for p in parts if p["class"] in ("brace_long", "brace_short")]
    wheels = [p for p in parts if p["class"] == "wheel"]
    xs = [p["center"][0] for p in parts]
    if braces:
        rear_x = sum(p["center"][0] for p in braces) / len(braces)
        front_x = max(xs) if rear_x < (min(xs) + max(xs)) / 2 else min(xs)
    else:
        rear_x, front_x = min(xs), max(xs)
    rear_sign = 1.0 if rear_x > front_x else -1.0  # +1 means rear is at larger x

    def toward_rear(p) -> float:
        return rear_sign * p["center"][0]

    # resolve magenta discs: wheel nut if within 30 mm of a wheel centre, else pin clip if within 22 mm of a pin
    pins = [p for p in parts if p["class"] == "pin"]
    for p in parts:
        if p["class"] != "disc":
            continue
        pc = Vector(p["center"])
        if any((pc - Vector(w["center"])).length < 30 for w in wheels):
            p["class"] = "wheel_nut"
        elif any((pc - Vector(q["center"])).length < 22 for q in pins):
            p["class"] = "pin_clip"
        else:
            p["class"] = "other"

    # pins: vertical (long axis z) vs horizontal
    for p in pins:
        d = p["dims"]
        p["orientation"] = "vertical" if d[2] >= max(d[0], d[1]) else "horizontal"

    # components
    knobs = [p for p in parts if p["class"] == "knob"]
    horizontal = sorted([p for p in pins if p["orientation"] == "horizontal"], key=toward_rear)
    vertical = [p for p in pins if p["orientation"] == "vertical"]
    comp_of_pin = {}
    if len(horizontal) >= 2:
        comp_of_pin[id(horizontal[-1])] = 6  # nearest the rear end
        comp_of_pin[id(horizontal[-2])] = 5
    elif len(horizontal) == 1:
        comp_of_pin[id(horizontal[0])] = None  # ambiguous: resolved by the caller from the state code
    for p in vertical:
        pc = Vector(p["center"])
        has_knob = any((pc - Vector(k["center"])).length < 30 for k in knobs)
        comp_of_pin[id(p)] = 2 if has_knob else 8  # the front chassis pin carries the thumb knob; the other vertical pin is the bracket screw
    for p in pins:
        p["component"] = comp_of_pin.get(id(p))
    for p in parts:
        cls = p["class"]
        pc = Vector(p["center"])
        if cls == "pin_clip":
            owner = min(pins, key=lambda q: (pc - Vector(q["center"])).length)
            p["component"] = owner.get("component")
        elif cls in ("wheel", "wheel_nut", "axle", "pulley"):
            mid = (rear_x + front_x) / 2
            p["component"] = 10 if (p["center"][0] - mid) * rear_sign > 0 else 9
        elif cls == "brace_long":
            p["component"] = 3
        elif cls == "brace_short":
            p["component"] = 4
        elif cls == "plate":
            p["component"] = 1
        elif cls == "knob":
            p["component"] = 2
        elif cls == "block":
            p["component"] = 9 if p["colour"] == "Magenta" else 7  # magenta block carries the front axle; white block is the bracket
        elif cls == "beam":
            p["component"] = 0
        elif cls == "pin":
            pass
        else:
            p["component"] = 0 if p["colour"] == "Metallic silver" and p["dims"][2] > 20 else None
    return parts


# ----------------------------------------------------------------------------------------------
# standalone: colour-coded sheet
# ----------------------------------------------------------------------------------------------

def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = 960, 600
    scene.render.image_settings.file_format = "JPEG"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    bpy.ops.import_scene.fbx(filepath=str(args.fbx))
    for ob in list(bpy.data.objects):
        if ob.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(ob, do_unlink=True)
    shells = split_shells([o for o in bpy.data.objects if o.type == "MESH"])
    parts = classify_file(shells)

    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in shells:
        l, h = world_bounds(ob)
        lo = Vector(min(a, b) for a, b in zip(lo, l))
        hi = Vector(max(a, b) for a, b in zip(hi, h))
    center = (lo + hi) / 2
    radius = (hi - lo).length / 2
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(50)
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    el = math.radians(60)
    cam.location = center + Vector((radius * 0.5, -radius * 2.3 * math.cos(el), radius * 2.3 * math.sin(el)))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    for i, pos in enumerate([(1, -1, 2), (-1, 1, 2)]):
        ld = bpy.data.lights.new(f"L{i}", "SUN")
        ld.energy = 2.0
        lo_ = bpy.data.objects.new(f"L{i}", ld)
        bpy.context.collection.objects.link(lo_)
        lo_.location = center + Vector(pos) * radius * 3
        lo_.rotation_euler = (center - lo_.location).to_track_quat("-Z", "Y").to_euler()

    mats = {}
    for cls, rgb in CLASS_COLOURS.items():
        m = bpy.data.materials.new(f"cls_{cls}")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = 0.6
        mats[cls] = m
    for ob, p in zip(shells, parts):
        m = mats.get(p["class"], mats["other"])
        for j in range(len(ob.data.materials)):
            ob.data.materials[j] = m
    scene.render.filepath = str(out / "classes.jpg")
    bpy.ops.render.render(write_still=True)
    (out / "parts.json").write_text(json.dumps(parts, indent=1))
    summary = {}
    for p in parts:
        summary.setdefault(p["class"], []).append(p.get("component"))
    for cls, comps in sorted(summary.items()):
        print(f"{cls:12s} n={len(comps):2d} components={comps}")


if __name__ == "__main__":
    main()
