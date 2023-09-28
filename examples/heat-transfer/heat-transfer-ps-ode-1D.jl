using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf
using OrdinaryDiffEq

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1);
else
    @init_parallel_stencil(Threads, Float64, 1);
end

@parallel function diffusion1D_step!(dT, T, Ci, lam, _dx)
    @inn(dT) = lam*@inn(Ci)*(@d2(T)*_dx^2);
    return
end

mutable struct KineticMonteCarlo
    t::Float64
    dt::Float64
    T
    const Ci
    const lam::Float64
    const dx::Float64
    const ℓ::Float64
end

function KineticMonteCarlo(ℓ::Real, dx::Real, dt::Real, Ci0::Real, lam::Real; 
                           T0::Real = 0.0)
    ni = length(0:dx:ℓ)
    T = @zeros(ni)
    T .= T0
    Ci = @zeros(ni)
    Ci .= Ci0
    KineticMonteCarlo(0.0, dt, T, Ci, lam, dx, ℓ)
end

function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    function f(dT, T, p, t)
        @parallel diffusion1D_step!(dT, T, kmc.Ci, kmc.lam, 1/kmc.dx)
        bc!(T)
    end
    prob = ODEProblem(f, kmc.T, (0.0, kmc.dt))
    sol = solve(prob, ROCK2(), save_everystep=false, save_start=false)
                #reltol=1e-8, abstol=1e-8)
    kmc.T = sol[end]
    Δt = -log(rand()) * kmc.dt
    kmc.t += Δt
end

function main1(; doplot::Bool=true)
    A = 1.0     # cm^2
    ℓ = 5.0     # cm
    a = 1.0     # cm/s^(1/2)
    h = 0.025   # cm
    xs = 0:h:ℓ
    ni = length(xs)
    T0 = 100.0
    Tb = 200.0
    dt = 1e-3
    analyt_T(x, t) = (Tb-T0)*erfc(x / (2*a*sqrt(t))) + T0

    kmc = KineticMonteCarlo(ℓ, h, dt, a, 1.0; T0=T0)
    @show size(kmc.T)

    bc!(T) = (T[1] = Tb; T[end] = T[end-1])
    bc!(kmc.T)

    idx = 1
    t_outs = [0.3; 0.7; 1.3; 2.0]
    iter = 0
    p = (doplot) ? plot() : nothing
    while (kmc.t <= 1.5)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, bc!)
        if (kmc.t >= t_outs[idx])
            Tavg = sum(kmc.T) / length(kmc.T)
            @show iter, t_outs[idx], kmc.t, Tavg
            if doplot
                p = scatter!(xs, kmc.T; label="\$t = $(t_outs[idx])\$, kMC")
                p = plot!(xs, map(x -> analyt_T(x, t_outs[idx]), xs); label="\$t = $(t_outs[idx])\$, exact")
                xlabel!(L"$x$")
                ylabel!(L"$T$")
            end
            idx += 1
        end
    end

    if doplot
        println("Enter to quit")
        display(p)
        readline()
    end

    kmc
end

if false # profile case 1
    Profile.clear()
    @time main1(; doplot=false)
    @profile main1(; doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
    println("=================================================================")
    println("=================================================================")
    println("=================================================================")
end

println("Implementation of the pure heat transfer models in section III.A");
println("using ParallelStencil")
println("===================================================================");

println("1D heat transfer using ParallelStencil.jl...");
@time main1(; doplot=true)
