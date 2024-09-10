import Pkg

for pkg in ["ParallelStencil", "StaticArrays", "PhysicalConstants", "Unitful",
            "ArgParse", "LaTeXStrings", "Plots", "PyPlot", "SpecialFunctions",
            "Profile", "PProf", "OrdinaryDiffEq", "Sundials", "MPI", 
            "Distributions", "ODEInterfaceDiffEq"]
    @show pkg
    Pkg.add(pkg)
end

Pkg.update()
