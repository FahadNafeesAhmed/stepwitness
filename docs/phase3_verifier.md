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

Version 2 is kept, and the first end-to-end run uses it, because a measured weak point beats an
unmeasured strong one.
