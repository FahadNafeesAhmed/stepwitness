"""Render randomized views of the IndustReal state meshes with labels. Runs inside Blender.

For every requested state file this script builds a fresh scene: the state mesh on a table, a few
lights with random strength and colour, a camera sampled on a hemisphere above the assembly looking
down at head-camera angles, loose distractor parts scattered on the table, and in some frames a crude
hand occluder. It writes one image per sample plus a JSON label with the state code, the 2D box of the
assembly, the camera pose and the lighting it used.

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_render_states.py -- \
        --fbx-dir data/industreal/part_geometries/part_geometries --out data/renders/v0 \
        --states 1,5,13 --per-state 4 --seed 0 --size 1280 720 --engine eevee

The state code for stateN comes from stepwitness.procedure.labelled_states_industreal().
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import labelled_states_industreal, state_str  # noqa: E402

SKIN = [(0.80, 0.60, 0.48), (0.55, 0.38, 0.28), (0.93, 0.78, 0.66), (0.36, 0.24, 0.18)]
TABLE_COLOURS = [(0.92, 0.92, 0.90), (0.75, 0.75, 0.73), (0.55, 0.42, 0.30), (0.25, 0.25, 0.27), (0.85, 0.80, 0.70)]


# ----------------------------------------------------------------------------------------------
# scene construction
# ----------------------------------------------------------------------------------------------

def reset_scene(engine: str, size: tuple[int, int], samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if engine == "eevee":
        # Blender 4.x calls the current EEVEE "BLENDER_EEVEE_NEXT"; 5.x dropped the suffix
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
    else:
        scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    if engine == "eevee":
        scene.eevee.taa_render_samples = samples
    else:
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 4
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        import os

        wanted = os.environ.get("STEPWITNESS_CYCLES_BACKEND", "CUDA")  # OPTIX fails to load its kernel on this driver
        prefs.compute_device_type = wanted
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == wanted
        print("cycles devices:", [(d.name, d.type, d.use) for d in prefs.devices])
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True


def import_state(fbx: Path) -> list:
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    # the files carry dozens of cameras and lights from their original export; drop them
    for ob in list(bpy.data.objects):
        if ob.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(ob, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for ob in meshes:
        ob["is_assembly"] = True
    return meshes


def world_bbox(objects) -> tuple[Vector, Vector]:
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in objects:
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector(min(a, b) for a, b in zip(lo, w))
            hi = Vector(max(a, b) for a, b in zip(hi, w))
    return lo, hi


def snapshot_material_colors() -> dict:
    out = {}
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                out[mat.name] = tuple(node.inputs["Base Color"].default_value)
    return out


def jitter_materials(rng: random.Random, originals: dict, gain: float = 0.15, offset: float = 0.02) -> None:
    """Multiplicative brightness jitter plus a small additive term, so dark parts stay dark instead
    of picking up a random hue (an additive jitter alone turned the black wheels olive and purple)."""
    for mat in bpy.data.materials:
        if mat.name not in originals or not mat.use_nodes:
            continue
        g = 1.0 + rng.uniform(-gain, gain)
        o = rng.uniform(-offset, offset)
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                c = list(originals[mat.name])
                for i in range(3):
                    c[i] = min(1.0, max(0.0, c[i] * g + o))
                node.inputs["Base Color"].default_value = c
                node.inputs["Roughness"].default_value = rng.uniform(0.35, 0.75)


GENERATED_MATERIALS = ("TableMat", "Silver", "MagentaPart", "WhitePart", "Skin")


def clear_generated() -> None:
    """Remove everything that is not the imported assembly, so the next sample starts clean
    without re-importing the FBX."""
    for ob in list(bpy.data.objects):
        if not ob.get("is_assembly"):
            bpy.data.objects.remove(ob, do_unlink=True)
    for coll, keep in ((bpy.data.lights, ()), (bpy.data.cameras, ())):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)
    for mat in list(bpy.data.materials):
        if mat.name.split(".")[0] in GENERATED_MATERIALS:
            bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_material(name: str, rgb, roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_table(rng: random.Random, lo: Vector, hi: Vector, extent: float):
    bpy.ops.mesh.primitive_plane_add(size=extent, location=((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z - 0.5))
    plane = bpy.context.active_object
    plane.name = "Table"
    base = rng.choice(TABLE_COLOURS)
    rgb = tuple(min(1.0, max(0.0, v + rng.uniform(-0.08, 0.08))) for v in base)
    mat = make_material("TableMat", rgb, roughness=rng.uniform(0.4, 0.9))
    # procedural grain so the table is not a flat colour
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = rng.uniform(0.002, 0.02)
    noise.inputs["Detail"].default_value = 4.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[v * 0.85 for v in rgb], 1.0)
    ramp.color_ramp.elements[1].color = (*rgb, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    plane.data.materials.append(mat)
    return plane


def add_lights(rng: random.Random, center: Vector, radius: float) -> list[dict]:
    specs = []
    n = rng.randint(2, 4)
    for i in range(n):
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(math.radians(25), math.radians(80))
        d = radius * rng.uniform(3.0, 6.0)
        pos = center + Vector((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)))
        kind = rng.choice(["AREA", "POINT", "SUN"])
        light_data = bpy.data.lights.new(f"Light{i}", kind)
        energy = {"AREA": rng.uniform(5e4, 4e5), "POINT": rng.uniform(5e4, 5e5), "SUN": rng.uniform(0.5, 2.5)}[kind]
        light_data.energy = energy
        temp = rng.uniform(0.85, 1.0)
        light_data.color = (1.0, temp, temp * rng.uniform(0.85, 1.0))
        if kind == "AREA":
            light_data.size = radius * rng.uniform(1.0, 3.0)
        ob = bpy.data.objects.new(f"Light{i}", light_data)
        bpy.context.collection.objects.link(ob)
        ob.location = pos
        direction = center - pos
        ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        specs.append({"kind": kind, "energy": energy, "position": [round(v, 1) for v in pos]})
    world = bpy.context.scene.world
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*[rng.uniform(0.2, 0.9)] * 3, 1.0)
    bg.inputs["Strength"].default_value = rng.uniform(0.1, 0.8)
    bpy.context.scene.view_settings.exposure = rng.uniform(-0.8, 0.3)
    return specs


def add_camera(rng: random.Random, center: Vector, radius: float, size: tuple[int, int]) -> dict:
    cam_data = bpy.data.cameras.new("Camera")
    hfov = math.radians(rng.uniform(52, 72))
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.lens_unit = "FOV"
    cam_data.angle = hfov
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    # head-camera geometry: above the table, looking down between 30 and 75 degrees
    az = rng.uniform(0, 2 * math.pi)
    el = rng.uniform(math.radians(30), math.radians(75))
    d = radius * rng.uniform(1.8, 4.2)
    pos = center + Vector((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)))
    # look near the assembly, not exactly at it, so it is not always centred
    target = center + Vector((rng.uniform(-radius, radius), rng.uniform(-radius, radius), 0.0)) * 0.8
    cam.location = pos
    cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(rng.uniform(-8, 8)))
    return {"position": [round(v, 1) for v in pos], "target": [round(v, 1) for v in target], "hfov_deg": round(math.degrees(hfov), 1)}


def add_distractors(rng: random.Random, lo: Vector, hi: Vector, radius: float) -> int:
    """Loose pins, washers and nuts on the table, as in the real recordings."""
    n = rng.randint(0, 10)
    silver = make_material("Silver", (0.75, 0.75, 0.78), roughness=0.35)
    magenta = make_material("MagentaPart", (0.75, 0.10, 0.45), roughness=0.6)
    white = make_material("WhitePart", (0.92, 0.92, 0.92), roughness=0.6)
    for i in range(n):
        kind = rng.choice(["pin", "washer", "nut", "brace"])
        x = rng.uniform(lo.x - 3 * radius, hi.x + 3 * radius)
        y = rng.uniform(lo.y - 3 * radius, hi.y + 3 * radius)
        if abs(x - (lo.x + hi.x) / 2) < radius * 0.9 and abs(y - (lo.y + hi.y) / 2) < radius * 0.9:
            continue  # keep the assembly clear
        if kind == "pin":
            bpy.ops.mesh.primitive_cylinder_add(radius=6, depth=rng.choice([25, 50]), location=(x, y, lo.z + 6))
            ob = bpy.context.active_object
            ob.rotation_euler = (math.pi / 2, 0, rng.uniform(0, math.pi))
            ob.data.materials.append(silver)
        elif kind == "washer":
            bpy.ops.mesh.primitive_torus_add(major_radius=9, minor_radius=3, location=(x, y, lo.z + 3))
            ob = bpy.context.active_object
            ob.data.materials.append(rng.choice([silver, white]))
        elif kind == "nut":
            bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=10, depth=8, location=(x, y, lo.z + 4))
            ob = bpy.context.active_object
            ob.data.materials.append(rng.choice([silver, white]))
        else:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, lo.z + 3))
            ob = bpy.context.active_object
            ob.scale = (rng.uniform(40, 100), 12, 6)
            ob.rotation_euler = (0, 0, rng.uniform(0, math.pi))
            ob.data.materials.append(rng.choice([magenta, white]))
        ob.name = f"Distractor{i}"
    return n


def add_hand(rng: random.Random, center: Vector, radius: float) -> bool:
    """A crude hand: a palm ellipsoid and a few finger capsules, placed over part of the assembly."""
    if rng.random() > 0.45:
        return False
    skin = make_material("Skin", rng.choice(SKIN), roughness=0.8)
    side = rng.uniform(0, 2 * math.pi)
    reach = radius * rng.uniform(0.3, 1.1)
    base = center + Vector((reach * math.cos(side), reach * math.sin(side), radius * rng.uniform(0.3, 0.9)))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=base)
    palm = bpy.context.active_object
    palm.scale = (radius * 0.55, radius * 0.4, radius * 0.18)
    palm.rotation_euler = (0, 0, side)
    palm.data.materials.append(skin)
    palm.name = "HandPalm"
    for f in range(rng.randint(3, 5)):
        ang = side + math.pi + math.radians(-30 + 15 * f)
        tip = base + Vector((radius * 0.7 * math.cos(ang), radius * 0.7 * math.sin(ang), -radius * 0.05 * f))
        bpy.ops.mesh.primitive_cylinder_add(radius=radius * 0.07, depth=radius * 0.6, location=(base + tip) / 2)
        finger = bpy.context.active_object
        finger.rotation_euler = (tip - base).to_track_quat("Z", "Y").to_euler()
        finger.data.materials.append(skin)
        finger.name = f"HandFinger{f}"
    return True


# ----------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------

def project_bbox(objects, size: tuple[int, int]) -> list[float] | None:
    """2D box (x, y, w, h in pixels) of the assembly's mesh vertices, clipped to the image."""
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore

    scene = bpy.context.scene
    cam = scene.camera
    xs, ys = [], []
    for ob in objects:
        mesh = ob.data
        step = max(1, len(mesh.vertices) // 1500)
        for i in range(0, len(mesh.vertices), step):
            co = world_to_camera_view(scene, cam, ob.matrix_world @ mesh.vertices[i].co)
            if co.z <= 0:
                continue
            xs.append(co.x)
            ys.append(1.0 - co.y)
    if not xs:
        return None
    w, h = size
    x0, x1 = max(0.0, min(xs)) * w, min(1.0, max(xs)) * w
    y0, y1 = max(0.0, min(ys)) * h, min(1.0, max(ys)) * h
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def render_state(state_no: int, fbx: Path, out: Path, n: int, seed: int, size, engine, samples, states) -> list[dict]:
    records = []
    code = state_str(states[state_no])
    reset_scene(engine, size, samples)
    meshes = import_state(fbx)
    originals = snapshot_material_colors()
    lo, hi = world_bbox(meshes)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1.0)
    for k in range(n):
        rng = random.Random(seed * 100003 + state_no * 1009 + k)
        clear_generated()
        jitter_materials(rng, originals)
        add_table(rng, lo, hi, extent=radius * 30)
        lights = add_lights(rng, center, radius)
        cam = add_camera(rng, center, radius, size)
        n_distractors = add_distractors(rng, lo, hi, radius)
        hand = add_hand(rng, center, radius)
        bbox = project_bbox(meshes, size)
        if bbox is None:
            continue  # assembly out of frame; skip this sample
        name = f"state{state_no:02d}_{seed:03d}_{k:04d}"
        img_path = (out / "images" / f"{name}.jpg").resolve()  # Blender resolves relative paths against the drive root
        img_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(img_path)
        bpy.ops.render.render(write_still=True)
        rec = {
            "image": img_path.name,
            "state_no": state_no,
            "state": code,
            "bbox_xywh": bbox,
            "camera": cam,
            "lights": lights,
            "distractors": n_distractors,
            "hand": hand,
            "engine": engine,
            "size": list(size),
        }
        records.append(rec)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "labels" / f"{name}.json").write_text(json.dumps(rec))
    return records


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", default="all", help="comma list of state numbers 1..22, or all")
    ap.add_argument("--per-state", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args(argv)

    states = labelled_states_industreal()
    wanted = list(range(1, 23)) if args.states == "all" else [int(s) for s in args.states.split(",")]
    all_records = []
    for s in wanted:
        fbx = args.fbx_dir / f"state{s}.fbx"
        recs = render_state(s, fbx, args.out, args.per_state, args.seed, tuple(args.size), args.engine, args.samples, states)
        all_records.extend(recs)
        print(f"state{s}: rendered {len(recs)} / {args.per_state}")
    manifest = args.out / f"manifest_seed{args.seed:03d}.json"
    manifest.write_text(json.dumps(all_records, indent=1))
    print(f"wrote {manifest} with {len(all_records)} records")


if __name__ == "__main__":
    main()
