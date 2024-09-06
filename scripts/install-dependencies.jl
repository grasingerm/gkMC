import Pkg

for pkg in ["ParallelStencil", "StaticArrays", "PhysicalConstants", "Unitful",
            "ArgParse", "LaTeXStrings", "Plots", "PythonPlot", "SpecialFunctions",
            "Profile", "PProf", "OrdinaryDiffEq", "Sundials", "MPI", 
            "Distributions"]
    @show pkg
    Pkg.add(pkg)
end

Pkg.update()
