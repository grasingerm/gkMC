# gkMC

Playing around and trying to learn some kinetic Monte Carlo.
I'm interested in nonisothermal problems in which chemical reactions, phase transitions, latent heat, etc., are all relevant.
As a result, I've been following the paper "Kinetic Monte Carlo modeling of chemical reactions coupled with heat transfer" (DOI: [10.1063/1.2877443](http://dx.doi.org/10.1063/1.2877443)).
All examples are written in julia.

## Pure heat transfer

The pure heat transfer problems outlined in section III.A are implemented in ``heat-transfer.jl``.

## Heated desorption of a lattice gas

The next example involve the heated desorption of a lattice gas (section IV).
The model is implemented in ``gas-desorption.jl``.
When the enthalpy change is neglected (command line argument ``--dT``), the simulation appears to agree with Fig 4.
To see other arguments, run

    julia gas-desorption.jl --help

## ParallelStencil.jl

``ParallelStencil.jl`` is a julia package which can be used for highly efficient, parallel simulation of heat transfer.
I have used it in ``heat-transfer-parallel-stencil.jl`` and ``gas-desorption-ps.jl``.
Both simulations appear to be orders of magnitude faster than their analogs.
