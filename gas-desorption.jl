using Distributed
@everywhere using StaticArrays
@everywhere import Base.CartesianIndex
@everywhere import PhysicalConstants
@everywhere using Unitful
using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf

@everywhere const _kB = PhysicalConstants.CODATA2018.BoltzmannConstant
@everywhere const _NA = PhysicalConstants.CODATA2018.AvogadroConstant

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
    gas::Array{Bool, 3}
    const logA::Real
    const EA::Real
    const h::Real
    const offset::Point
end

@everywhere function KineticMonteCarlo(dims::Point, logA::Real, EA::Real, h::Real; 
                           T0::Real = 0.0, offset::Point=Point(0.0, 0.0, 0.0))
    ni, nj, nk = map(i -> length(0.0:h:dims[i]), 1:3)
    ret = KineticMonteCarlo(0.0, fill(T0, ni, nj, nk), fill(true, ni, nj, nk), 
                            logA, EA, h, offset)
    ret.gas[:, 1, 1] .= false;
    return ret;
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

function kmc_events(kmc::KineticMonteCarlo, tbt::ThermalBitTrans;
                    active::Array{Bool, 3} = fill(true, size(kmc.T)))
    events = []
    c = tbt.a^2 / (kmc.h^2 * tbt.δT)
    ni, nj, nk = size(kmc.T)
    nn = ni*nj*nk
    np = nprocs()
    pp = round(Int, nn / np)
    idx_ranges = [( (i-1)*pp + 1):( i*pp ) for i=1:(np-1) ]
    if length(idx_ranges) > 0
        push!(idx_ranges, (idx_ranges[end].stop+1):nn)
    else
        push!(idx_ranges, 1:nn)
    end
    events = vcat(pmap(idx_range -> begin
        local_events = []
        for idx in idx_range
            J = vidx_to_Idx(idx, ni, nj, nk)
            ciJ = CartesianIndex(J)
            @inbounds Tj = kmc.T[ciJ]
            @inbounds if kmc.gas[ciJ]
                push!(local_events, (
                                     f=adsorp!, args=(J,), 
                                     rate=exp(-kmc.EA/Tj + kmc.logA)
                                    )
                     )
            end
            for nbr in nnbrs
                K = J + nbr
                @inbounds if (1 <= K[1] <= ni && 1 <= K[2] <= nj && 1 <= K[3] <= nk 
                              && active[CartesianIndex(K)])
                    @inbounds Tk = kmc.T[CartesianIndex(K)]
                    if Tj > Tk
                        rate = c * (Tj - Tk)
                        push!(local_events, (f=transfer_heat_fromto!, 
                                             args=(tbt, J, K), 
                                             rate=rate))
                    elseif Tj < Tk
                        rate = c * (Tk - Tj)
                        push!(local_events, (f=transfer_heat_fromto!, 
                                             args=(tbt, K, J), 
                                             rate=rate))
                    end
                end
            end
        end
        local_events
    end, idx_ranges)... )
end

@everywhere function transfer_heat_fromto!(kmc::KineticMonteCarlo, 
                                           tbt::ThermalBitTrans,
                                           from::Idx, to::Idx)
    kmc.T[CartesianIndex(to)] += tbt.δT
    kmc.T[CartesianIndex(from)] -= tbt.δT
end

@everywhere function adsorp!(kmc::KineticMonteCarlo, J::Idx)
    kmc.gas[CartesianIndex(J)] = false
end

function do_event!(kmc::KineticMonteCarlo, tbt::ThermalBitTrans)
    events = kmc_events(kmc, tbt)
    rate_cumsum = cumsum(map(event -> event.rate, events))
    choice_dec = rand()*rate_cumsum[end]
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    f, args, rate = events[choice_idx]
    f(kmc, args...)
    Δt = -log(rand()) / rate_cumsum[end]
    kmc.t += Δt
end

function main1(; maxiter::Int=Int(1.5*10^6), iterout::Int=100, 
                 iterplot::Int=(2*maxiter),
                 maxtime::Real=10^5, doplot::Bool=true)
    logA = log(10)                # log(ps^-1)
    @show EA = uconvert(Unitful.NoUnits, 10.8*10^3*u"J / mol" / _kB / _NA / 1u"K")  # 1 / K
    ni = 1001
    nj = 102
    a = sqrt(2.0)                 # nm/ps^(1/2)
    h = 2.0                       # nm
    ℓ = h*(ni-1)                  # nm
    d = h*(nj-1)                  # nm
    δT = 0.2                      # K
    xs = 0:h:ℓ
    ys = 0:h:d
    T0 = 100.0                    # K
    Tb = 110.0                    # K

    tbt = ThermalBitTrans(a, h, δT)
    kmc = KineticMonteCarlo(Point(ℓ, d, 0.0), logA, EA, h; T0=T0)
    @show size(kmc.T)
    @show N_active_sites = sum(kmc.gas)

    bc!(kmc::KineticMonteCarlo) = (kmc.T[:, 1, :] .= Tb)
    bc!(kmc)

    t_series = [];
    T_series = [];
    α_series = [];

    bc!(kmc)
    idx = 1
    iter = 0
    last_update = time()

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, tbt)
        bc!(kmc)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(α_series, sum(kmc.gas) / N_active_sites)
        end
        if (iter % iterplot == 0)
            p = heatmap(kmc.T[:, :, 1])
            title!("Temperature")
            println("Enter to quit")
            display(p)
            readline()
            p = heatmap(kmc.gas[:, :, 1])
            title!("Gas")
            println("Enter to quit")
            display(p)
            readline()
        end
        if (time() - last_update > 15)
            last_update = time()
            t = kmc.t
            Tavg = sum(kmc.T) / length(kmc.T)
            α = sum(kmc.gas) / N_active_sites
            @show iter, t, α, Tavg, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.gas) / N_active_sites)

    if doplot
        p = plot(t_series, α_series)
        xlabel!("time")
        ylabel!("gas %")
        println("Enter to quit")
        display(p)
        readline()
        p = plot(t_series, T_series)
        xlabel!("time")
        ylabel!("avg T")
        println("Enter to quit")
        display(p)
        readline()
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

println("My implementation of the heated desorption of a lattice gas, neglecting enthalpy change; section IV.A");
println("===================================================================");
@time main1(; doplot=true)
