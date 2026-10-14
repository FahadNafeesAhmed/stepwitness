# Phase 2: the render factory

## What the dataset ships

`part_geometries.zip` holds one FBX per labelled correct state (`state1.fbx` to `state22.fbx`), four
3MF files with the loose parts by category, and a state overview sheet. Each state FBX is a single
merged mesh, exported at its own level of detail, with dozens of stray cameras and lights that are
deleted on import. Materials are named by colour family (White, Magenta, Metallic silver, Black).

## Two approaches that were tried and dropped

**Mapping parts across files.** The plan was to identify each component once in the complete model
and derive every other state by deleting groups. `tools/blender_derive_states.py` rebuilds all 22
states that way and renders each beside the dataset's own file. The pairs do not match for states
1 to 12: the complete model is modelled differently at the front end from the early-state files, and
shells do not line up across files even after anchoring on a shared part. Rejected.

**Matching shells by geometry across files.** Vertex counts differ per file because of the per-file
decimation, and positions drift by more than a few millimetres. Rejected.

## What is used

**Per-file classification by geometry.** `tools/blender_classify_parts.py` separates a state mesh
into its loose shells and labels each shell by colour and size: wheel, wheel nut, axle, pin, pin clip,
knob, long brace, short brace, plate, block, beam, pulley. It then assigns components from the car's
layout: braces mark the rear end, horizontal pins near the braces are the rear pins ordered along the
car, the vertical pin on the plates is the front chassis pin, the vertical pin with the thumb knob is
the bracket screw, the white block is the bracket, the magenta block carries the front axle, wheels
and their nuts belong to the rear or front assembly by which end they are nearer. Plates and short
braces share dimensions and are told apart by orientation. The assignment was checked against the
states where parts first appear: state 8 has one vertical pin and no knob, state 9 adds the block,
the second vertical pin and the knob together.

**Rendering from each state's own file.** `tools/blender_render_dataset.py` imports a state once,
then per sample rebuilds the scene: table with procedural grain, two to four lights of random kind,
strength and warmth, exposure jitter, a camera on a hemisphere at 30 to 78 degrees looking down and
2 to 4.2 car-radii away with a 55 to 72 degree field of view, copies of the file's own parts scattered
on the table as distractors, an instruction sheet in some frames, and a crude hand in some frames.
Material colours were set to the real footage (matte grey pins, bright magenta, white PLA, black
wheels) and jittered multiplicatively so dark parts stay dark.

**Error generation.** With a chosen fraction, one component present in the state is edited:
pin pulled part-way out, clips removed, pin turned across, pin swapped for a longer or shorter one,
wheel nuts removed, wheel slid outward, braces slid or tilted, plates slid, bracket turned, screw
discs removed. The label is the state code with -1 at that component, which is the dataset's own
convention for an incorrectly installed component.

## Machine facts that cost time

- EEVEE shader compilation and Cycles both fail when the GPU is short of memory. A local language
  model holding 5 GB of the 8 GB card was the cause; unload it before rendering.
- OptiX kernels fail to load on this driver; Cycles runs on CUDA.
- Blender resolves a relative output path against the drive root, not the working directory.
- `bpy_extras.object_utils.world_to_camera_view` reads a stale camera matrix until the view layer is
  updated; half the samples were silently skipped before that call was added.
- EEVEE at 720p with 16 samples renders at about two frames per second once the file is imported.

## Output

`data/renders/v2/`: `images/*.jpg`, `manifest_seed*.json` with per-image state code, label code,
error component and kind, box, camera, lights, distractor count and hand flag. The verifier and the
localizer are built from these manifests.
