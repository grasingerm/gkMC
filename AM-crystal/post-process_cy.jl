using Glob
using DelimitedFiles

σ(x, c, a) = 1 / (1 + exp(-a*(x - c)))

if length(ARGS) < 1
    @info("usage: julia post-process_cg.jl DATADIR")
    exit(-1)
end
datadir = ARGS[1]
a = (length(ARGS) > 1) ? ARGS[2] : 10.0

cgs = map(datafile -> begin
        iter = split(split(datafile, "-")[end], ".")[1]
        crystalfile = joinpath(datadir, "crystal-$iter.csv")
        
        cgdata = readdlm(datafile, ',')
        ni, nj = size(cgdata)
        for i=1:ni, j=1:nj
            @show i, j, "do something!"
        end
        nactive = sum(readdlm(crystalfile, ',') .!= 0)
        return cgtotal / nactive
    end, (filter(x -> contains(x, "cg-crystal") && endswith(x, ".csv"), readdir(datadir; join=true))))

println(maximum(cgs))
