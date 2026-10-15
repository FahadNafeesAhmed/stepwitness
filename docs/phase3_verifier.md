# Phase 3: the verifier

## Version 2: global embedding, 144-way head

Built on 2026-10-14 from 13,200 renders (22 correct states, 122 error classes), split 80/20 by
sample so the validation renders are new views of the same objects. Frozen DINOv2-base, crops at
224 pixels with a 15 percent margin and 8 percent box jitter, CLS and mean-patch pooled to 1,536
dimensions. Prototype bank: 16 centroids per class. Head: two-layer MLP, 40 epochs.

| Verifier | 144-way accuracy | Correct state identified | Error states named | Error detection precision | Error detection recall | False alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| prototypes only | 0.226 | 0.471 | 0.071 | 0.532 | 0.422 | 0.322 |
| head only | 0.383 | 0.623 | 0.068 | 0.865 | 0.110 | 0.015 |
| blended | 0.375 | 0.613 | 0.062 | 0.834 | 0.112 | 0.019 |

Read it plainly: the verifier names the correct state about six times in ten, and it almost never
names the error. The head learned to say "correct" and be safe. The prototypes catch more errors
but with a false alarm on a third of correct crops.

## Why, and what changes

The errors are small. A clip is 3 mm across; a pin pulled 10 mm out moves by a few pixels. The crop
is the whole car, about 150 mm, resized to 224 pixels, and then pooled to one vector. A 3 mm clip is
four pixels wide in that image and gone after pooling. The states that differ by one pin (2, 3, 4; 6,
7, 8; 14, 15; 19, 20, 21) are where most of the state confusions sit, for the same reason.

The class design makes it worse. Each of the 122 error classes has 21 to 60 renders, split further
across nine error kinds, so the head sees a handful of examples of "clip missing on pin 6 at state 8".
But the label the dataset uses is per component: a −1 at position k. "Component k is wrong" is the
same target across every state that contains k, and there are hundreds of renders of it.

Version 3 therefore changes three things:

1. **Two heads instead of 144 classes.** A 22-way head for the correct state and an 11-way
   sigmoid head for "component k is installed wrongly", trained together on the same features. The
   recognizer consumes exactly that pair: a state posterior and per-component error probabilities.
2. **Patch features instead of one pooled vector.** The backbone's patch tokens are kept as a grid
   and a small attention-pooling head reads them, so a few patches that show a clip can dominate.
3. **Higher input resolution.** 448 pixels instead of 224, so the clip is eight pixels, not four.

Version 2 is kept as the measured weak point.

## Version 3: patch tokens at 448 px, state head and component-error head

Built on 2026-10-14 from the same 13,200 renders and the same 80/20 split. Frozen DINOv2-base at
448 pixels, patch tokens pooled to an 8 x 8 grid plus the CLS token, an attention-pooling head with
four queries, a 22-way state output and an 11-way component-error output trained together, 30 epochs.

| Measure on held-out renders | v2 | v3 |
|---|---:|---:|
| Correct state identified | 0.623 | 0.916 |
| Exact label (state and every error digit) | 0.383 | 0.625 |
| Component error: precision | | 0.641 |
| Component error: recall | | 0.472 |
| Component error: false-positive rate per component | | 0.012 |
| Crop has some error: precision | 0.865 | 0.773 |
| Crop has some error: recall | 0.110 | 0.541 |
| Crop has some error: false-alarm rate | 0.015 | 0.138 |

Reading: the state is now right nine times in ten, and about half of the small errors are seen in a
single crop. The recognizer sees many crops of the same state, so what it can catch over a few
seconds is higher than the single-crop recall; what that comes to on real footage is the number the
monitor run gives.

Cost of the change: tokens are 65 x 768 per crop instead of a single 1,536 vector, and the backbone
runs at 448 pixels, roughly four times the compute of 224. On this GPU that is still well inside a
10 fps budget.

The three build failures before this result were all the machine running out of memory with the
author's other applications open; the fixes that made the build survivable were a memory-mapped
token cache, chunked copies to the GPU, and resumable extraction.
