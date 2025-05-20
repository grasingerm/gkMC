using LatinHypercubeSampling;
using Distributed;
@everywhere using Printf;

@everywhere fmt(x) = @sprintf("%07d", round(Int, 1e3*x));
@everywhere fmt_int(x) = @sprintf("%03d", x);

@everywhere function prefix(T0, Tbed, EA, J, Jm, M, σ, B)
    "T0-$(fmt(T0))_Tbed-$(fmt(Tbed))_EA-$(fmt(EA))_J-$(fmt(J))_Jm-$(fmt(Jm))_M-$(fmt(M))_sigma-$(fmt(σ))_B-$(fmt(B))"
end

workdir = if length(ARGS) > 0
    ARGS[1]
else
    @info "Usage: julia lhs_study.jl workdir <nsamples> <ngens>"
    exit(-1)
end

nsamples = if length(ARGS) > 1
    parse(Int, ARGS[2])
else
    256
end

ngens = if length(ARGS) > 2
    parse(Int, ARGS[3])
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
         outdir = joinpath(workdir, prefix(T0, Tbed, EA, J, Jm, M, σ, B))
         if !isdir(outdir)
             println("Running case: ($T0, $Tbed, $EA, $J, $Jm, $M, $σ, $B)")
             command = `julia -t 1 -O 3 AM-crystal-Potts.jl --trow $trow --v0 $v0 --lx $lx --ly $ly --ni $ni --nj $nj --T0 $T0 --Tbed $Tbed --EA $EA --J $J --Jm $Jm --M $M --sigma-init $σ --B $B --outdir $outdir`;
             output = read(command, String);
             write(joinpath(outdir, "stdout.txt"), output); 
         else
             println("Case: ($T0, $Tbed, $EA, $J, $Jm, $M, $σ, $B) has already been run.")
         end
     end, 1:nsamples)
