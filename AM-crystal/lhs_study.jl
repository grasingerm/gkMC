using LatinHypercubeSampling;
using Distributed;
@everywhere using Printf;

@everywhere fmt(x) = @sprintf("%07d", round(Int, 1e3*x));
@everywhere fmt_int(x) = @sprintf("%03d", x);

@everywhere function prefix(T0, Tbed, EA, J, Jm, M, σ, B)
    error("Not yet implemented")
    "Tbed-$(fmt(case[:E0]))_K1-$(fmt(case[:K1]))_K2-$(fmt(case[:K2]))_kT-$(fmt(case[:kT]))_Fz-$(fmt(case[:Fz]))_Fx-$(fmt(case[:Fx]))_n-$(fmt(case[:n]))_b-$(fmt(case[:b]))_kappa-$(fmt(case[:kappa]))_run-$(case[:run])";
end

workdir = if length(ARGS) > 0
    ARGS[1]
else
    @info "Usage: julia lhs_study.jl workdir <nsamples>"
    exit(-1)
end

nsamples = if length(ARGS) > 1
    convert(Int, ARGS[2])
else
    256
end

ngens = if length(ARGS) > 2
    convert(Int, ARGS[3])
else
    7
end

EA = 1316.0
T0s = [673.0, 693.0]
Tbeds = [393, 418, 448]
v0 = 0.8
lx = 8
ni = 401
trow = 0.65e-1;
ly = 3*trow;
nj = 16

# EA, J, Jm, M, sigma-init, B
ranges = [
          (0.25*EA, 2.5*EA),
          (0.0, 10*EA),
          (0.0, 10*EA),
          (0.0, 1000),
          (0.0, 10),
          (0.0, 2.5*EA)
         ]

ndims = length(ranges)

plan, _ = LHCoptim(nsamples, ndims, ngens)
scaled_plan = scaleLHC(plan, ranges)

pmap(i -> begin
         T0 = T0s[1]
         Tbed = Tbeds[1]
         EA, J, Jm, M, σ, B = scaled_plan[i, :]
         outdir = joinpath(workdir, prefix(J, Jm, M, σ, B))
         if !isdir(outdir)
             println("Running case: ($T0, $Tbed, $EA, $J, $Jm, $M, $σ, $B)")
             command = `julia -t 1 -O 3 AM-crystal-Potts.jl `;
         else
             println("Case: ($T0, $Tbed, $EA, $J, $Jm, $M, $σ, $B) has already been run.")
         end
     end, 1:nsamples)
