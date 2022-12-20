@everywhere using StaticArrays
@everywhere import Base.CartesianIndex
using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf
using Distributed

@everywhere abstract type HeatTrans; end

κ_to_a(κ::Real, σ::Real, ρ::Real) = sqrt(κ / (σ*ρ))

@everywhere struct ThermalBitTrans <: HeatTrans
    a::Real
    h::Real
    δT::Real
    #σρ::Real
end

@everywhere struct PoissonTrans <: HeatTrans
    a::Real
end

@everywhere Point = SVector{3, Float64};

@everywhere mutable struct KineticMonteCarlo
    t::Real
    T::Array{Float64, 3}
    const h::Real
    const offset::Point
end

@everywhere function KineticMonteCarlo(dims::Point, h::Real; 
                           T0::Real = 0.0, offset::Point=Point(0.0, 0.0, 0.0))
    ni, nj, nk = map(i -> length(0.0:h:dims[i]), 1:3)
    KineticMonteCarlo(0.0, fill(T0, ni, nj, nk), h, offset)
end

@everywhere function ijk_to_pos(i::Int, j::Int, k::Int; h::Real = 1.0, 
                                offset::Point=Point(0.0, 0.0, 0.0))
    offset + h*Point(i-1, j-1, k-1)
end

@everywhere Idx = SVector{3, Int}
@everywhere CartesianIndex(idx::Idx) = CartesianIndex(Tuple(idx))
@everywhere nnbrs = [
                        Idx(1, 0, 0),
                        Idx(0, 1, 0),
                        Idx(0, 0, 1),
                    ]
@everywhere function vidx_to_Idx(idx::Int, ni::Int, nj::Int, nk::Int)
    i = ((idx-1) % ni) + 1
    j = (((idx-1) ÷ ni) % nj) + 1
    k = ((idx-1) ÷ (ni*nj)) + 1
    Idx(i, j, k)
end

function heat_transfer_events(kmc::KineticMonteCarlo, tbt::ThermalBitTrans;
                              active::Array{Bool, 3} = fill(true, size(kmc.T)))
    events = []
    c = tbt.a^2 / (kmc.h^2 * tbt.δT)
    ni, nj, nk = size(kmc.T)
    nn = ni*nj*nk
    np = nprocs()
    pp = round(Int, nn / np)
    idx_ranges = [( (i-1)*pp + 1):( i*pp ) for i=1:(np-1) ]
    push!(idx_ranges, (idx_ranges[end].stop+1):nn)
    events = vcat(pmap(idx_range -> begin
        local_events = []
        for idx in idx_range
            J = vidx_to_Idx(idx, ni, nj, nk)
            @inbounds Tj = kmc.T[CartesianIndex(J)]
            for nbr in nnbrs
                K = J + nbr
                @inbounds if (1 <= K[1] <= ni && 1 <= K[2] <= nj && 1 <= K[3] <= nk 
                              && active[CartesianIndex(K)])
                    @inbounds Tk = kmc.T[CartesianIndex(K)]
                    if Tj > Tk
                        rate = c * (Tj - Tk)
                        push!(local_events, (from=J, to=K, rate=rate))
                    elseif Tj < Tk
                        rate = c * (Tk - Tj)
                        push!(local_events, (from=K, to=J, rate=rate))
                    end
                end
            end
        end
        local_events
    end, idx_ranges)... )
end

function transfer_heat!(kmc::KineticMonteCarlo, tbt::ThermalBitTrans)
    events = heat_transfer_events(kmc, tbt)
    rate_cumsum = cumsum(map(event -> event.rate, events))
    choice_dec = rand()*rate_cumsum[end]
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    J, K, rate = events[choice_idx]
    kmc.T[CartesianIndex(K)] += tbt.δT
    kmc.T[CartesianIndex(J)] -= tbt.δT
    Δt = -log(rand()) / rate_cumsum[end]
    kmc.t += Δt
end

function main1(; doplot::Bool=true)
    A = 1.0   # cm^2
    ℓ = 5.0   # cm
    a = 1.0   # cm/s^(1/2)
    h = 0.025  # cm
    δT = 0.2  # K
    xs = 0:h:ℓ
    ni = length(xs)
    T0 = 100.0
    Tb = 200.0
    analyt_T(x, t) = (Tb-T0)*erfc(x / (2*a*sqrt(t))) + T0

    @show tbt = ThermalBitTrans(a, h, δT)
    @show kmc = KineticMonteCarlo(Point(ℓ, 0.0, 0.0), h; T0=T0)
    @show size(kmc.T)

    bc!(kmc::KineticMonteCarlo) = (kmc.T[1, 1, 1] = Tb)
    bc!(kmc)
    @show heat_transfer_events(kmc, tbt)

    bc!(kmc)
    idx = 1
    t_outs = [0.3; 0.7; 1.3; 2.0]
    iter = 0
    p = (doplot) ? plot() : nothing
    while (kmc.t <= 1.5)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, tbt)
        bc!(kmc)
        if (kmc.t >= t_outs[idx])
            Tavg = sum(kmc.T) / length(kmc.T)
            @show iter, t_outs[idx], kmc.t, Tavg
            if doplot
                p = scatter!(xs, vec(kmc.T); label="\$t = $(t_outs[idx])\$, kMC")
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

    kmc, tbt
end

function main2(; maxiter=Inf, doplot::Bool=true)
    th = 1.0   # cm
    ℓ = 5.0    # cm
    a = 1.0    # cm/s^(1/2)
    h = 0.2    # cm
    δT = 0.3   # K
    xs = 0:h:ℓ
    ni = length(xs)
    nj = ni
    @show Xs = repeat(xs, 1, nj)
    @show Ys = repeat(reshape(xs, 1, nj), ni, 1)
    T0 = 200.0
    Tb = 100.0
    analyt_T(x, y, t) = (T0-Tb)*erf(x / (2*a*sqrt(t)))*erf(y / (2*a*sqrt(t))) + Tb

    @show tbt = ThermalBitTrans(a, h, δT)
    @show kmc = KineticMonteCarlo(Point(ℓ, ℓ, 0.0), h; T0=T0)
    @show size(kmc.T)

    bc!(kmc::KineticMonteCarlo) = (kmc.T[:, 1, 1] .= Tb; kmc.T[1, :, 1] .= Tb;)
    bc!(kmc)
    @show kmc.T
    @show heat_transfer_events(kmc, tbt)

    bc!(kmc)
    idx = 1
    t_outs = [0.3; 0.8; Inf]
    iter = 0
    last_updated = time()
    while (kmc.t <= 0.9 && iter <= maxiter)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, tbt)
        bc!(kmc)
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

    kmc, tbt
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

if false # profile case 2
    Profile.clear()
    @time main2(; maxiter=10, doplot=false)
    @profile main2(; maxiter=1000, doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
end

println("My implementation of the pure heat transfer models in section III.A");
println("===================================================================");

@time main1(; doplot=false)
@time main2(; doplot=false)
