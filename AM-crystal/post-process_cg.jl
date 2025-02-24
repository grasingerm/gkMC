using Glob
using DelimitedFiles

if length(ARGS) < 1
    @info("usage: julia post-process_cg.jl DATADIR")
    exit(-1)
end
datadir = ARGS[1]

cgs = map(datafile -> begin
        iter = split(split(datafile, "-")[end], ".")[1]
        crystalfile = joinpath(datadir, "crystal-$iter.csv")

        cgtotal = sum(readdlm(datafile, ','))
        nactive = sum(readdlm(crystalfile, ',') .!= 0)
        return cgtotal / nactive
    end, (filter(x -> contains(x, "cg-crystal") && endswith(x, ".csv"), readdir(datadir; join=true))))

println(maximum(cgs))
