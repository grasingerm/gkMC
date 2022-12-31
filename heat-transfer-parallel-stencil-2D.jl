using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel function diffusion2D_step!(T2, T, Ci, lam, dt, _dx, _dy)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2));
    return
end

mutable struct KineticMonteCarlo
    t::Float64
    dt::Float64
    T
    const Ci
    const lam::Float64
    const dx::Float64
    const ℓx::Float64
    const dy::Float64
    const ℓy::Float64
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real, 
                           dt::Real, Ci0::Real, lam::Real; 
                           T0::Real = 0.0)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    T = @zeros(ni, nj)
    T .= T0
    Ci = @zeros(ni, nj)
    Ci .= Ci0
    KineticMonteCarlo(0.0, dt, T, Ci, lam, dx, ℓx, dy, ℓy)
end

function transfer_heat!(kmc::KineticMonteCarlo, T2, bc!::Function)
    @parallel diffusion2D_step!(T2, kmc.T, kmc.Ci, kmc.lam, kmc.dt, 1/kmc.dx, 1/kmc.dy)
    bc!(T2)
    kmc.T, T2 = T2, kmc.T
    Δt = -log(rand()) * kmc.dt
    kmc.t += Δt
end

function main2(; maxiter=Inf, doplot::Bool=true)
    th = 1.0   # cm
    ℓ = 5.0    # cm
    a = 1.0    # cm/s^(1/2)
    h = 0.05    # cm
    δT = 0.3   # K
    xs = 0:h:ℓ
    ni = length(xs)
    nj = ni
    @show Xs = repeat(xs, 1, nj)
    @show Ys = repeat(reshape(xs, 1, nj), ni, 1)
    T0 = 200.0
    Tb = 100.0
    dt = 1e-6
    analyt_T(x, y, t) = (T0-Tb)*erf(x / (2*a*sqrt(t)))*erf(y / (2*a*sqrt(t))) + Tb

    kmc = KineticMonteCarlo(ℓ, h, ℓ, h, dt, a, 1.0; T0=T0)
    T2 = @zeros(ni, nj)
    T2[:, :] = kmc.T[:, :]
    @show size(kmc.T)

    bc!(T) = (T[:, end] = T[:, end-1]; T[end, :] = T[end-1, :]; 
              T[:, 1] .= Tb; T[1, :] .= Tb;)
    bc!(kmc.T)
    
    idx = 1
    t_outs = [0.3; 0.8; Inf]
    iter = 0
    last_updated = time()
    while (kmc.t <= 0.9 && iter <= maxiter)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, T2, bc!)
        if (kmc.t >= t_outs[idx])
            Tavg = sum(kmc.T) / length(kmc.T)
            @show iter, t_outs[idx], kmc.t, Tavg
            if doplot
                p = scatter(vec(Xs), vec(Ys), vec(kmc.T); label="\$t = $(t_outs[idx])\$, kMC")
                zs = zeros(ni, nj);
                for i=1:ni, j=1:nj
                    zs[i, j] = analyt_T(xs[i], xs[j], t_outs[idx]);
                end
                p = plot!(xs, xs, zs; st=:surf, label="\$t = $(t_outs[idx])\$, exact")
                xlabel!(L"$x$")
                ylabel!(L"$y$")
                zlabel!(L"$T$")
                println("Enter to continue...")
                display(p)
                readline()
            end
            idx += 1
        end
        if time() - last_updated > 10
            Tavg = sum(kmc.T) / length(kmc.T)
            last_updated = time()
            @show iter, kmc.t, Tavg, last_updated
        end
    end

    kmc
end

if false # profile case 2
    Profile.clear()
    @time main2(; maxiter=10, doplot=false)
    @profile main2(; maxiter=1000, doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
end

println("Implementation of the pure heat transfer models in section III.A");
println("using ParallelStencil")
println("===================================================================");

println("2D heat transfer using ParallelStencil.jl...");
@time main2(; doplot=true)
