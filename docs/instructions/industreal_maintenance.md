# Work instruction: construction-toy car, maintenance

Reconstructed from the IndustReal paper (figure 1, section 4.1) and its labelled states. The task is a
component upgrade: replace the rear chassis with long braces by the rear chassis with short braces.
The graph in `stepwitness/procedures/industreal_maintenance.json` cites these line numbers.

The rear wheels and the two rear pins may be removed in any order. The rear chassis comes off only
once both pins and the rear wheels are off, which means lines 2, 3 and 4 done. The new pins go in
after the short chassis, in either order. The rear wheels go back on last, after both new pins,
lines 7 and 8. Line 1 describes the starting condition and is not a step.

1. Start from the fully assembled car.
2. Remove the rear wheel assembly.
3. Pull the front-rear chassis pin.
4. Pull the rear-rear chassis pin.
5. Lift off the rear chassis with the long braces.
6. Fit the short rear chassis onto the base.
7. Secure the short rear chassis with the front-rear chassis pin.
8. Secure the short rear chassis with the rear-rear chassis pin.
9. Reinstall the rear wheel assembly.
