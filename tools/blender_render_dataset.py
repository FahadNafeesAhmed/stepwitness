"""Render the training set: randomized views of every correct state and of error states built by
editing parts inside each state file. Runs inside Blender.

Compared with the first renderer this one matches the materials to the real footage (matte grey
pins, bright magenta, white PLA), scatters copies of the real parts on the table as distractors,
adds an instruction sheet, and can generate execution errors on any component present in a state.

Error types, applied to one component of the state (the label gets -1 at that component):
  pin_out          pin and its clips pulled part-way out along the pin axis
  pin_unsecured    one or both clips of a pin removed
  pin_rotated      horizontal pin lying across instead of through
  wrong_pin        pin replaced by one 1.5x or 0.6x as long
  wheel_no_nut     a wheel's nuts removed
  wheel_loose      a wheel slid outward along its axle
  brace_shifted    braces slid along the car or tilted
  plate_shifted    front plates slid along the car
  bracket_rotated  bracket block turned by a quarter turn
  screw_missing    bracket screw discs removed

Usage (from the repo root):
    third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_render_dataset.py -- \
        --fbx-dir data/industreal/part_geometries/part_geometries --out data/renders/v2 \
        --states all --per-state 300 --error-fraction 0.5 --seed 0 --size 1280 720
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from blender_classify_parts import classify_file, split_shells, world_bounds  # noqa: E402
from stepwitness.procedure import labelled_states_industreal, state_str  # noqa: E402

# colours measured by eye from the real recordings (white desk, office light): matte PLA
REAL_COLOURS = {
    "Magenta": (0.80, 0.12, 0.42),
    "Metallic silver": (0.42, 0.43, 0.45),
    "White": (0.90, 0.90, 0.88),
    "Black": (0.03, 0.03, 0.03),
}
SKIN = [(0.80, 0.60, 0.48), (0.55, 0.38, 0.28), (0.93, 0.78, 0.66), (0.36, 0.24, 0.18), (0.70, 0.50, 0.40)]
TABLE_COLOURS = [(0.93, 0.93, 0.92), (0.88, 0.88, 0.86), (0.80, 0.80, 0.78), (0.70, 0.68, 0.64), (0.55, 0.42, 0.30)]
DISTRACTOR_CLASSES = ("pin", "pin_clip", "wheel_nut", "wheel", "brace_long", "brace_short", "plate", "knob")
ERRORS_BY_CLASS = {
    "pin": ["pin_out", "pin_unsecured", "pin_rotated", "wrong_pin"],
    "wheel": ["wheel_no_nut", "wheel_loose"],
    "brace_long": ["brace_shifted"],
    "brace_short": ["brace_shifted"],
    "plate": ["plate_shifted"],
    "block": ["bracket_rotated"],
}


# ----------------------------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------------------------

def reset_scene(engine: str, size, samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if engine == "eevee":
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
        scene.eevee.taa_render_samples = samples
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "CUDA"
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True


def import_state(fbx: Path) -> list:
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    for ob in list(bpy.data.objects):
        if ob.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(ob, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    return split_shells(meshes)


def set_real_materials() -> dict:
    """Replace the file's material colours by the measured real ones; return originals for jitter."""
    originals = {}
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        family = mat.name.split(".")[0]
        rgb = REAL_COLOURS.get(family)
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                if rgb is not None:
                    node.inputs["Base Color"].default_value = (*rgb, 1.0)
                node.inputs["Roughness"].default_value = 0.65
                if "Specular IOR Level" in node.inputs:
                    node.inputs["Specular IOR Level"].default_value = 0.3
                originals[mat.name] = tuple(node.inputs["Base Color"].default_value)
    return originals


def jitter_materials(rng: random.Random, originals: dict, gain: float = 0.15, offset: float = 0.03) -> None:
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
                node.inputs["Roughness"].default_value = rng.uniform(0.45, 0.85)


GENERATED = "generated"


def tag_generated(ob) -> None:
    ob[GENERATED] = True


def make_material(name: str, rgb, roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    mat[GENERATED] = True
    return mat


def clear_generated() -> None:
    for ob in list(bpy.data.objects):
        if ob.get(GENERATED):
            bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.lights, bpy.data.cameras, bpy.data.meshes):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)
    for mat in list(bpy.data.materials):
        if mat.get(GENERATED):
            bpy.data.materials.remove(mat)


def add_table(rng: random.Random, lo: Vector, hi: Vector, extent: float):
    bpy.ops.mesh.primitive_plane_add(size=extent, location=((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z - 0.3))
    plane = bpy.context.active_object
    tag_generated(plane)
    base = rng.choice(TABLE_COLOURS)
    rgb = tuple(min(1.0, max(0.0, v + rng.uniform(-0.06, 0.06))) for v in base)
    mat = make_material("TableMat", rgb, roughness=rng.uniform(0.5, 0.95))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = rng.uniform(0.002, 0.03)
    noise.inputs["Detail"].default_value = 3.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*[v * rng.uniform(0.85, 0.97) for v in rgb], 1.0)
    ramp.color_ramp.elements[1].color = (*rgb, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    plane.data.materials.append(mat)


def add_paper(rng: random.Random, lo: Vector, hi: Vector, radius: float) -> None:
    """An A4-like instruction sheet lying on the table beside the assembly, with dark print."""
    if rng.random() > 0.7:
        return
    side = rng.choice([-1, 1])
    cx = (lo.x + hi.x) / 2 + side * rng.uniform(radius * 1.6, radius * 3.0)
    cy = (lo.y + hi.y) / 2 + rng.uniform(-radius, radius)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, lo.z + 0.2))
    paper = bpy.context.active_object
    tag_generated(paper)
    paper.scale = (210 * rng.uniform(0.8, 1.1), 297 * rng.uniform(0.8, 1.1), 1)
    paper.rotation_euler = (0, 0, rng.uniform(-0.5, 0.5))
    mat = make_material("Paper", (0.95, 0.95, 0.94), roughness=0.9)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexVoronoi")
    tex.inputs["Scale"].default_value = rng.uniform(6, 14)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.1, 0.1, 0.12, 1.0)
    ramp.color_ramp.elements[1].position = rng.uniform(0.25, 0.45)
    ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.94, 1.0)
    links.new(tex.outputs["Distance"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    paper.data.materials.append(mat)


def add_lights(rng: random.Random, center: Vector, radius: float) -> list[dict]:
    specs = []
    for i in range(rng.randint(2, 4)):
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(math.radians(30), math.radians(85))
        d = radius * rng.uniform(3.0, 6.0)
        pos = center + Vector((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)))
        kind = rng.choice(["AREA", "AREA", "POINT", "SUN"])
        ld = bpy.data.lights.new(f"Light{i}", kind)
        energy = {"AREA": rng.uniform(5e4, 3.5e5), "POINT": rng.uniform(5e4, 4e5), "SUN": rng.uniform(0.4, 2.0)}[kind]
        ld.energy = energy
        warm = rng.uniform(0.88, 1.0)
        ld.color = (1.0, warm, warm * rng.uniform(0.88, 1.0))
        if kind == "AREA":
            ld.size = radius * rng.uniform(1.5, 4.0)
        ob = bpy.data.objects.new(f"Light{i}", ld)
        bpy.context.collection.objects.link(ob)
        tag_generated(ob)
        ob.location = pos
        ob.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()
        specs.append({"kind": kind, "energy": round(energy), "position": [round(v, 1) for v in pos]})
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*[rng.uniform(0.3, 0.9)] * 3, 1.0)
    bg.inputs["Strength"].default_value = rng.uniform(0.15, 0.7)
    bpy.context.scene.view_settings.exposure = rng.uniform(-0.6, 0.3)
    return specs


def add_camera(rng: random.Random, center: Vector, radius: float) -> dict:
    cam_data = bpy.data.cameras.new("Camera")
    hfov = math.radians(rng.uniform(55, 72))
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.lens_unit = "FOV"
    cam_data.angle = hfov
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    tag_generated(cam)
    bpy.context.scene.camera = cam
    az = rng.uniform(0, 2 * math.pi)
    el = rng.uniform(math.radians(30), math.radians(78))
    d = radius * rng.uniform(2.0, 4.2)
    pos = center + Vector((d * math.cos(el) * math.cos(az), d * math.cos(el) * math.sin(az), d * math.sin(el)))
    target = center + Vector((rng.uniform(-radius, radius), rng.uniform(-radius, radius), 0.0)) * 0.45
    cam.location = pos
    cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(rng.uniform(-10, 10)))
    return {"position": [round(v, 1) for v in pos], "target": [round(v, 1) for v in target], "hfov_deg": round(math.degrees(hfov), 1)}


def add_distractors(rng: random.Random, shells: list, parts: list[dict], lo: Vector, hi: Vector, radius: float) -> int:
    """Copies of the file's own parts scattered on the table, as in the real recordings."""
    candidates = [(ob, p) for ob, p in zip(shells, parts) if p["class"] in DISTRACTOR_CLASSES]
    if not candidates:
        return 0
    n = rng.randint(3, 22)
    placed = 0
    for i in range(n):
        src, p = rng.choice(candidates)
        for _ in range(8):
            x = rng.uniform(lo.x - 3.5 * radius, hi.x + 3.5 * radius)
            y = rng.uniform(lo.y - 3.5 * radius, hi.y + 3.5 * radius)
            if abs(x - (lo.x + hi.x) / 2) > radius * 1.05 or abs(y - (lo.y + hi.y) / 2) > radius * 1.05:
                break
        else:
            continue
        copy = src.copy()
        copy.data = src.data.copy()
        bpy.context.collection.objects.link(copy)
        tag_generated(copy)
        copy["is_assembly"] = False
        slo, shi = world_bounds(src)
        scenter = (slo + shi) / 2
        dims = shi - slo
        # lay the part flat: rotate so its shortest axis points up, then random yaw
        axes = sorted(range(3), key=lambda k: dims[k])
        rot = Matrix.Identity(4)
        if axes[0] == 0:
            rot = Matrix.Rotation(math.radians(90), 4, "Y")
        elif axes[0] == 1:
            rot = Matrix.Rotation(math.radians(90), 4, "X")
        yaw = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, "Z")
        height = dims[axes[0]]
        copy.matrix_world = Matrix.Translation(Vector((x, y, lo.z + height / 2))) @ yaw @ rot @ Matrix.Translation(-scenter) @ src.matrix_world
        placed += 1
    return placed


def add_hand(rng: random.Random, center: Vector, radius: float) -> bool:
    if rng.random() > 0.45:
        return False
    skin = make_material("Skin", rng.choice(SKIN), roughness=0.85)
    side = rng.uniform(0, 2 * math.pi)
    reach = radius * rng.uniform(0.5, 1.3)
    base = center + Vector((reach * math.cos(side), reach * math.sin(side), radius * rng.uniform(0.2, 0.6)))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=base)
    palm = bpy.context.active_object
    tag_generated(palm)
    palm.scale = (radius * 0.36, radius * 0.28, radius * 0.12)
    palm.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), side)
    palm.data.materials.append(skin)
    for f in range(rng.randint(3, 5)):
        ang = side + math.pi + math.radians(-30 + 15 * f)
        tip = base + Vector((radius * 0.65 * math.cos(ang), radius * 0.65 * math.sin(ang), -radius * 0.06 * f))
        bpy.ops.mesh.primitive_cylinder_add(radius=radius * 0.065, depth=radius * 0.55, location=(base + tip) / 2)
        finger = bpy.context.active_object
        tag_generated(finger)
        finger.rotation_euler = (tip - base).to_track_quat("Z", "Y").to_euler()
        finger.data.materials.append(skin)
    return True


# ----------------------------------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------------------------------

def long_axis(p: dict) -> Vector:
    d = p["dims"]
    k = max(range(3), key=lambda i: d[i])
    v = Vector((0, 0, 0))
    v[k] = 1.0
    return v


def apply_error(rng: random.Random, shells: list, parts: list[dict], component: int, hidden: list) -> str | None:
    """Edit the shells of one component in place. Returns the error type, or None if not applicable.
    Objects to delete are hidden (moved far away) and listed in ``hidden`` for restoration."""
    members = [(ob, p) for ob, p in zip(shells, parts) if p.get("component") == component]
    if not members:
        return None
    classes = {p["class"] for _, p in members}
    options = []
    for cls in classes:
        options.extend(ERRORS_BY_CLASS.get(cls, []))
    if not options:
        return None
    kind = rng.choice(sorted(set(options)))

    def hide(ob):
        ob.location = ob.location + Vector((0, 0, 5000))
        hidden.append(ob)

    def group(cls):
        return [ob for ob, p in members if p["class"] == cls]

    if kind in ("pin_out", "pin_unsecured", "pin_rotated", "wrong_pin"):
        pins = [(ob, p) for ob, p in members if p["class"] == "pin"]
        pin, pp = rng.choice(pins)
        clips = [ob for ob, p in members if p["class"] == "pin_clip" and (Vector(p["center"]) - Vector(pp["center"])).length < 22]
        axis = long_axis(pp)
        if kind == "pin_out":
            shift = axis * rng.uniform(8, 16) * rng.choice([-1, 1])
            for ob in [pin] + clips:
                ob.location = ob.location + shift
        elif kind == "pin_unsecured":
            if not clips:
                return None
            for ob in rng.sample(clips, k=rng.randint(1, len(clips))):
                hide(ob)
        elif kind == "pin_rotated":
            c = Vector(pp["center"])
            rot = Matrix.Rotation(math.radians(90), 4, "Z" if axis.z < 0.5 else "X")
            for ob in [pin] + clips:
                ob.matrix_world = Matrix.Translation(c) @ rot @ Matrix.Translation(-c) @ ob.matrix_world
            for ob in clips:
                hide(ob)
        elif kind == "wrong_pin":
            c = Vector(pp["center"])
            s = rng.choice([1.5, 0.6])
            scale = Matrix.Scale(s, 4, axis)
            pin.matrix_world = Matrix.Translation(c) @ scale @ Matrix.Translation(-c) @ pin.matrix_world
            if s > 1:
                for ob in clips:
                    hide(ob)
    elif kind in ("wheel_no_nut", "wheel_loose"):
        wheels = [(ob, p) for ob, p in members if p["class"] == "wheel"]
        wheel, wp = rng.choice(wheels)
        nuts = [ob for ob, p in members if p["class"] == "wheel_nut" and (Vector(p["center"]) - Vector(wp["center"])).length < 30]
        if kind == "wheel_no_nut":
            if not nuts:
                return None
            for ob in nuts:
                hide(ob)
        else:
            axles = [p for _, p in members if p["class"] == "axle"]
            if axles:
                axis = long_axis(axles[0])
                origin = Vector(axles[0]["center"])
            else:
                # no axle shell classified in this file: the wheel's thin dimension is the axle direction
                # and the other wheel of the pair gives the inward direction
                d = wp["dims"]
                k = min(range(3), key=lambda i: d[i])
                axis = Vector((0, 0, 0))
                axis[k] = 1.0
                others = [p for _, p in members if p["class"] == "wheel" and p is not wp]
                origin = Vector(others[0]["center"]) if others else Vector(wp["center"]) - axis
            outward = axis if (Vector(wp["center"]) - origin).dot(axis) > 0 else -axis
            shift = outward * rng.uniform(8, 16)
            for ob in [wheel] + nuts:
                ob.location = ob.location + shift
    elif kind == "brace_shifted":
        braces = group("brace_long") + group("brace_short")
        if rng.random() < 0.5:
            shift = Vector((rng.uniform(10, 18) * rng.choice([-1, 1]), 0, 0))
            for ob in braces:
                ob.location = ob.location + shift
        else:
            for ob in braces:
                lo, hi = world_bounds(ob)
                c = (lo + hi) / 2
                rot = Matrix.Rotation(math.radians(rng.uniform(8, 16) * rng.choice([-1, 1])), 4, "Z")
                ob.matrix_world = Matrix.Translation(c) @ rot @ Matrix.Translation(-c) @ ob.matrix_world
    elif kind == "plate_shifted":
        shift = Vector((rng.uniform(10, 18) * rng.choice([-1, 1]), 0, 0))
        for ob in group("plate"):
            ob.location = ob.location + shift
    elif kind == "bracket_rotated":
        for ob in group("block"):
            lo, hi = world_bounds(ob)
            c = (lo + hi) / 2
            rot = Matrix.Rotation(math.radians(90), 4, "Z")
            ob.matrix_world = Matrix.Translation(c) @ rot @ Matrix.Translation(-c) @ ob.matrix_world
    elif kind == "screw_missing":
        clips = group("pin_clip")
        if not clips:
            return None
        for ob in clips:
            hide(ob)
    return kind


def error_candidates(parts: list[dict], state: tuple) -> list[int]:
    comps = set()
    for p in parts:
        c = p.get("component")
        if c is None or c == 0 or state[c] != 1:
            continue
        if ERRORS_BY_CLASS.get(p["class"]):
            comps.add(c)
    return sorted(comps)


# ----------------------------------------------------------------------------------------------
# labels and main loop
# ----------------------------------------------------------------------------------------------

def project_bbox(objects, size) -> list[float] | None:
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore

    bpy.context.view_layer.update()  # refresh matrix_world of the new camera and of moved parts
    scene = bpy.context.scene
    cam = scene.camera
    xs, ys = [], []
    for ob in objects:
        mesh = ob.data
        step = max(1, len(mesh.vertices) // 400)
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
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def render_state(state_no: int, fbx: Path, out: Path, n: int, seed: int, size, engine, samples, states, error_fraction: float) -> list[dict]:
    records = []
    state = states[state_no]
    code = state_str(state)
    reset_scene(engine, size, samples)
    shells = import_state(fbx)
    parts = classify_file(shells)
    # a single horizontal rear pin is ambiguous (5 or 6): take the one the state says is installed
    if state[5] + state[6] == 1:
        for p in parts:
            if p["class"] in ("pin", "pin_clip") and p.get("component") is None:
                p["component"] = 5 if state[5] == 1 else 6
    # vertical pins: the state code says which of the front chassis pin (2) and bracket screw (8) exist
    if state[2] == 1 and state[8] == 0:
        for p in parts:
            if p.get("component") == 8:
                p["component"] = 2
    elif state[2] == 0 and state[8] == 1:
        for p in parts:
            if p.get("component") == 2:
                p["component"] = 8
    originals = set_real_materials()
    original_matrices = [ob.matrix_world.copy() for ob in shells]
    lo, hi = Vector((math.inf,) * 3), Vector((-math.inf,) * 3)
    for ob in shells:
        l, h = world_bounds(ob)
        lo = Vector(min(a, b) for a, b in zip(lo, l))
        hi = Vector(max(a, b) for a, b in zip(hi, h))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1.0)
    candidates = error_candidates(parts, state)
    for k in range(n):
        rng = random.Random(seed * 100003 + state_no * 1009 + k)
        clear_generated()
        for ob, m in zip(shells, original_matrices):
            ob.matrix_world = m.copy()
        hidden: list = []
        error_comp, error_kind = None, None
        if candidates and rng.random() < error_fraction:
            error_comp = rng.choice(candidates)
            error_kind = apply_error(rng, shells, parts, error_comp, hidden)
            if error_kind is None:
                error_comp = None
        label_state = list(state)
        if error_comp is not None:
            label_state[error_comp] = -1
        jitter_materials(rng, originals)
        add_table(rng, lo, hi, extent=radius * 40)
        add_paper(rng, lo, hi, radius)
        lights = add_lights(rng, center, radius)
        cam = add_camera(rng, center, radius)
        n_distractors = add_distractors(rng, shells, parts, lo, hi, radius)
        hand = add_hand(rng, center, radius)
        visible = [ob for ob in shells if ob not in hidden]
        bbox = project_bbox(visible, size)
        if bbox is None:
            print(f"  skip state{state_no} sample {k}: assembly not in frame (camera {cam})")
            continue
        tag = "err" if error_comp is not None else "ok"
        name = f"s{state_no:02d}_{tag}_{seed:03d}_{k:04d}"
        img_path = (out / "images" / f"{name}.jpg").resolve()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(img_path)
        bpy.ops.render.render(write_still=True)
        rec = {
            "image": img_path.name,
            "state_no": state_no,
            "correct_state": code,
            "label_state": ",".join(str(v) for v in label_state),
            "error_component": error_comp,
            "error_kind": error_kind,
            "bbox_xywh": bbox,
            "camera": cam,
            "lights": lights,
            "distractors": n_distractors,
            "hand": hand,
            "size": list(size),
        }
        records.append(rec)
    return records


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", default="all")
    ap.add_argument("--per-state", type=int, default=100)
    ap.add_argument("--error-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args(argv)

    states = labelled_states_industreal()
    wanted = list(range(1, 23)) if args.states == "all" else [int(s) for s in args.states.split(",")]
    all_records = []
    args.out.mkdir(parents=True, exist_ok=True)
    for s in wanted:
        recs = render_state(s, args.fbx_dir / f"state{s}.fbx", args.out, args.per_state, args.seed, tuple(args.size), args.engine, args.samples, states, args.error_fraction)
        all_records.extend(recs)
        n_err = sum(1 for r in recs if r["error_component"] is not None)
        print(f"state{s}: rendered {len(recs)} ({n_err} with errors)")
        (args.out / f"manifest_seed{args.seed:03d}.json").write_text(json.dumps(all_records, indent=1))
    print(f"wrote manifest with {len(all_records)} records")


if __name__ == "__main__":
    main()
