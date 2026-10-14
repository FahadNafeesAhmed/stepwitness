# Work instruction: construction-toy car, assembly

Reconstructed from the IndustReal paper (figure 1, section 4.1) and its labelled states. The authors'
printed sheet is not published; this text is the input the procedure compiler reads, and the graph in
`stepwitness/procedures/industreal_assembly.json` cites these line numbers.

The front and rear subgoals may be built in either order. Within a subgoal, pins go in after the
part they secure, and the subgoal counts as complete only when its part and every one of its pins
are in. The bracket needs both subgoals complete, which means lines 3, 5 and 6 done. Front wheels
need the bracket and its screw. Rear wheels need both subgoals complete, lines 3, 5 and 6. Line 1
describes the starting condition and is not a step.

1. Start from the base.
2. Fit the front chassis onto the base.
3. Secure the front chassis with the front chassis pin.
4. Fit the rear chassis onto the base.
5. Secure the rear chassis with the front-rear chassis pin.
6. Secure the rear chassis with the rear-rear chassis pin.
7. Mount the front bracket.
8. Fix the front bracket with its screw.
9. Install the front wheel assembly.
10. Install the rear wheel assembly.
