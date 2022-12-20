# gkMC

Playing around and trying to learn some kinetic Monte Carlo.
I'm interested in nonisothermal problems in which chemical reactions, phase transitions, latent heat, etc., are all relevant.
As a result, I've been following the paper "Kinetic Monte Carlo modeling of chemical reactions coupled with heat transfer" (DOI: [10.1063/1.2877443](http://dx.doi.org/10.1063/1.2877443)).
All examples are written in julia.

## Pure heat transfer

The pure heat transfer problems outlined in section III.A are implemented in ``heat-transfer.jl``.

## Heated desorption of a lattice gas

The next examples involve the heated desorption of a lattice gas (section IV.A).
The first example neglects the enthalpy change from desorption and is implemented in ``gas-desorption.jl``.
It appears to agree with Fig 4.
