using Distributed
@everywhere using StaticArrays
@everywhere import Base.CartesianIndex
@everywhere import PhysicalConstants
@everywhere using Unitful
using ArgParse;
using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf
using DelimitedFiles

s = ArgParseSettings();
@add_arg_table! s begin
  "--dT", "-d"
    help = "change in temperature due to desorption"
    arg_type = Float64
    default = 0.0
  "--diff", "-a"
    help = "thermal diffusivity"
    arg_type = Float64
    default = sqrt(2.0)
  "--maxiter", "-m"
    help = "maximum number of iterations"
    arg_type = Int
    default = convert(Int, 1e7)
  "--maxtime", "-t"
    help = "maximum simulation time"
    arg_type = Float64
    default = 1e5
  "--figname"
    help = "figure name"
    arg_type = String
    default = "graph"
  "--showplot"
    help = "show plots"
    action = :store_true
end

pargs = parse_args(s);

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

@everywhere mutable struct PoissonTrans <: HeatTrans
    τc::Real
    const a::Real
    const max_τc::Real
end

@everywhere Point = SVector{3, Float64};

@everywhere mutable struct KineticMonteCarlo
    t::Real
    T::Array{Float64, 3}
    gas::Array{Bool, 3}
    const logA::Real
    const EA::Real
    const ΔT::Real
    const h::Real
    const offset::Point
end

@everywhere function KineticMonteCarlo(dims::Point, logA::Real, EA::Real, h::Real; 
                                       T0::Real = 0.0, 
                                       offset::Point=Point(0.0, 0.0, 0.0),
                                       ΔT::Real = 0.0
                                      )
    ni, nj, nk = map(i -> length(0.0:h:dims[i]), 1:3)
    ret = KineticMonteCarlo(0.0, fill(T0, ni, nj, nk), fill(true, ni, nj, nk), 
                            logA, EA, ΔT, h, offset)
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

function chunk_idx_ranges(nn::Int)
    np = nprocs()
    pp = round(Int, nn / np)
    idx_ranges = [( (i-1)*pp + 1):( i*pp ) for i=1:(np-1) ]
    if length(idx_ranges) > 0
        push!(idx_ranges, (idx_ranges[end].stop+1):nn)
    else
        push!(idx_ranges, 1:nn)
    end
    return idx_ranges
end

function kmc_events(kmc::KineticMonteCarlo, tbt::ThermalBitTrans;
                    active::Array{Bool, 3} = fill(true, size(kmc.T)))
    events = []
    c = tbt.a^2 / (kmc.h^2 * tbt.δT)
    ni, nj, nk = size(kmc.T)
    idx_ranges = chunk_idx_ranges(ni*nj*nk)
    
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

function kmc_events(kmc::KineticMonteCarlo, pht::PoissonTrans;
                    active::Array{Bool, 3} = fill(true, size(kmc.T)))
    events = []
    ni, nj, nk = size(kmc.T)
    idx_ranges = chunk_idx_ranges(ni*nj*nk)
    
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
        end
        local_events
    end, idx_ranges)... )
    if length(events) > 0
        pht.τc = min(sum(map(event -> event.rate, events)), pht.max_τc) # adaptive time integration
    end
    push!(events, (f=transfer_heat!, args=(pht,), rate=1 / pht.τc))

    return events
end


@everywhere function transfer_heat_fromto!(kmc::KineticMonteCarlo, 
                                           tbt::ThermalBitTrans,
                                           from::Idx, to::Idx)
    kmc.T[CartesianIndex(to)] += tbt.δT
    kmc.T[CartesianIndex(from)] -= tbt.δT
end

@everywhere function transfer_heat_3D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
    α = pht.a^2*pht.τc / (kmc.h^2)
    ni, nj, nk = size(kmc.T)
    new_T = zeros(ni, nj, nk)

    idx_ranges = chunk_idx_ranges((ni-2)*(nj-2)*(nk*2))
    new_inner_T = vcat(pmap(idx_range -> begin
        local_T = map(idx -> begin
            J = vidx_to_Idx(idx, ni-2, nj-2, nk-2) + Idx(1, 1, 1)
            isum = -6*kmc.T[CartesianIndex(J)]
            for nnbr in nnbrs
                @inbounds isum += (kmc.T[CartesianIndex(J + nnbr)] + 
                                   kmc.T[CartesianIndex(J - nnbr)] )
            end
            local_T[idx] = α*isum + kmc.T[CartesianIndex(J)]
        end, idx_range)
    end, idx_ranges)... )
    new_T[2:end-1, 2:end-1, 2:end-1] = reshape(new_inner_T, ni-2, nj-2, nk-2)

    for odim=1:3
        idims = collect(1:3)
        deleteat!(idims, odim)
        for i=2:(size(kmc.T, idims[1])-1), j=2:(size(kmc.T, idims[2])-1)
            K = i*nnbrs[idims[1]] + j*nnbrs[idims[2]] + nnbrs[odim]
            nbr_sum = -3*kmc.T[CartesianIndex(K)]
            for idim in idims
                nbr_sum += (kmc.T[CartesianIndex(K + nnbrs[idim])] + 
                            kmc.T[CartesianIndex(K - nnbrs[idim])])
            end
            nbr_sum += (kmc.T[CartesianIndex(K + 2*nnbrs[odim])] -
                        2*kmc.T[CartesianIndex(K + nnbrs[odim])])
            new_T[CartesianIndex(K)] = α*nbr_sum + kmc.T[CartesianIndex(K)]
            
            K = i*nnbrs[idims[1]] + j*nnbrs[idims[2]] + size(kmc.T, odim)*nnbrs[odim]
            nbr_sum = -3*kmc.T[CartesianIndex(K)]
            for idim in idims
                nbr_sum += (kmc.T[CartesianIndex(K + nnbrs[idim])] + 
                            kmc.T[CartesianIndex(K - nnbrs[idim])])
            end
            nbr_sum += (kmc.T[CartesianIndex(K - 2*nnbrs[odim])] -
                        2*kmc.T[CartesianIndex(K - nnbrs[odim])])
            new_T[CartesianIndex(K)] = α*nbr_sum + kmc.T[CartesianIndex(K)]
        end
    end

    # Corners
    for (i, isgn) in zip([1, ni], [1, -1]), (j, jsgn) in zip([1, nj], [1, -1]), (k, ksgn) in zip([1, nk], [1, -1])
        new_T[i, j, k] = α*(kmc.T[i+2*isgn, j, k] - 2*kmc.T[i+isgn, j, k] + 
                            kmc.T[i, j+2*jsgn, k] - 2*kmc.T[i, j+jsgn, k] +  
                            kmc.T[i, j, k+2*ksgn] - 2*kmc.T[i, j, k+ksgn] +
                            3*kmc.T[i, j, k]) + kmc.T[i, j, k]
    end

    kmc.T[:, :, :] = new_T[:, :, :]
end

@everywhere function transfer_heat_2D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
    α = pht.a^2*pht.τc / (kmc.h^2)
    ni, nj, nk = size(kmc.T)
    new_T = zeros(ni, nj, nk)

    idx_ranges = chunk_idx_ranges((ni-2)*(nj-2))
    new_inner_T = vcat(pmap(idx_range -> begin
        local_T = map(idx -> begin
            J = vidx_to_Idx(idx, ni-2, nj-2, 1) + Idx(1, 1, 0)
            isum = -4*kmc.T[CartesianIndex(J)]
            for nnbr in nnbrs[1:2]
                @inbounds isum += (kmc.T[CartesianIndex(J + nnbr)] + 
                                   kmc.T[CartesianIndex(J - nnbr)] )
            end
            α*isum + kmc.T[CartesianIndex(J)]
        end, idx_range)
    end, idx_ranges)... )
    new_T[2:end-1, 2:end-1, 1] = reshape(new_inner_T, ni-2, nj-2, 1)

    for odim=1:2
        idims = collect(1:2)
        deleteat!(idims, odim)
        for i=2:(size(kmc.T, idims[1])-1)
            K = i*nnbrs[idims[1]] + nnbrs[odim] + nnbrs[3]
            nbr_sum = -kmc.T[CartesianIndex(K)]
            for idim in idims
                nbr_sum += (kmc.T[CartesianIndex(K + nnbrs[idim])] + 
                            kmc.T[CartesianIndex(K - nnbrs[idim])])
            end
            nbr_sum += (kmc.T[CartesianIndex(K + 2*nnbrs[odim])] -
                        2*kmc.T[CartesianIndex(K + nnbrs[odim])])
            new_T[CartesianIndex(K)] = α*nbr_sum + kmc.T[CartesianIndex(K)]
            
            K = i*nnbrs[idims[1]] + size(kmc.T, odim)*nnbrs[odim] + nnbrs[3]
            nbr_sum = -kmc.T[CartesianIndex(K)]
            for idim in idims
                nbr_sum += (kmc.T[CartesianIndex(K + nnbrs[idim])] + 
                            kmc.T[CartesianIndex(K - nnbrs[idim])])
            end
            nbr_sum += (kmc.T[CartesianIndex(K - 2*nnbrs[odim])] -
                        2*kmc.T[CartesianIndex(K - nnbrs[odim])])
            new_T[CartesianIndex(K)] = α*nbr_sum + kmc.T[CartesianIndex(K)]
        end
    end

    # Corners
    for (i, isgn) in zip([1, ni], [1, -1]), (j, jsgn) in zip([1, nj], [1, -1])
        new_T[i, j, 1] = α*(kmc.T[i+2*isgn, j, 1] - 2*kmc.T[i+isgn, j, 1] + 
                             kmc.T[i, j+2*jsgn, 1] - 2*kmc.T[i, j+jsgn, 1] +  
                             2*kmc.T[i, j, 1]) + kmc.T[i, j, 1]
    end

    kmc.T[:, :, :] = new_T[:, :, :]
end

@everywhere function transfer_heat_1D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
    α = pht.a^2*pht.τc / (kmc.h^2)
    ni, nj, nk = size(kmc.T)
    new_T = zeros(ni, nj, nk)

    idx_ranges = chunk_idx_ranges((ni-2))
    new_inner_T = vcat(pmap(idx_range -> begin
        local_T = map(idx -> begin
            J = vidx_to_Idx(idx, ni-2, 1, 1) + Idx(1, 0, 0)
            isum = -2*kmc.T[CartesianIndex(J)]
            nnbr = nnbrs[1]
            isum += (kmc.T[CartesianIndex(J + nnbr)] + 
                     kmc.T[CartesianIndex(J - nnbr)] )
            α*isum + kmc.T[CartesianIndex(J)]
        end, idx_range)
    end, idx_ranges)... )
    new_T[2:(ni-1), 1, 1] = reshape(new_inner_T, ni-2, 1, 1)

    new_T[1, 1, 1] = (α*(kmc.T[3, 1, 1] - 2*kmc.T[2, 1, 1] + kmc.T[1, 1, 1]) +
                      kmc.T[1, 1, 1])
    new_T[ni, 1, 1] = (α*(kmc.T[ni-2, 1, 1] - 2*kmc.T[ni-1, 1, 1] + kmc.T[ni, 1, 1]) +
                       kmc.T[ni, 1, 1])

    kmc.T[:, :, :] = new_T[:, :, :]
end

@everywhere function transfer_heat!(kmc::KineticMonteCarlo, pht::PoissonTrans)
    ni, nj, nk = size(kmc.T)

    if (minimum(size(kmc.T)) > 2)
        transfer_heat_3D!(kmc, pht)
    elseif (size(kmc.T, 2) > 2) 
        transfer_heat_2D!(kmc, pht)
    else
        transfer_heat_1D!(kmc, pht)
    end
end

@everywhere function adsorp!(kmc::KineticMonteCarlo, J::Idx)
    kmc.gas[CartesianIndex(J)] = false
    kmc.T[CartesianIndex(J)] -= kmc.ΔT
end

function do_event!(kmc::KineticMonteCarlo, ht::HeatTrans)
    events = kmc_events(kmc, ht)
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

function main2(; maxiter::Int=Int(1e7), iterout::Int=100, 
                 iterplot::Int=(2*maxiter),
                 maxtime::Real=1e5, doplot::Bool=true, showplot::Bool=false,
                 figname::String="graph",
                 ΔT::Real=0.0,
                 A::Real=10.0, 
                 a::Real=1.0)
    logA = log(A)                # log(ps^-1)
    @show EA = uconvert(Unitful.NoUnits, 10.8*10^3*u"J / mol" / _kB / _NA / 1u"K")  # 1 / K
    ni = 101
    nj = 102
    h = 2.0                       # nm
    ℓ = h*(ni-1)                  # nm
    d = h*(nj-1)                  # nm
    δT = 0.2                      # K
    xs = 0:h:ℓ
    ys = 0:h:d
    T0 = 100.0                    # K
    Tb = 110.0                    # K
    τc = 1e-1
    max_τc = τc
    jbed = 1

    pht = PoissonTrans(a, τc, max_τc)
    kmc = KineticMonteCarlo(Point(ℓ, d, 0.0), logA, EA, h; T0=T0, ΔT=ΔT)
    kmc.gas[:, 1:jbed, 1] .= false
    kmc.T[:, 1:jbed, 1] .= Tb
    @show size(kmc.T)
    @show N_active_sites = sum(kmc.gas)

    bc!(kmc::KineticMonteCarlo) = (
                                   kmc.T[:, 1, :] .= Tb; 
                                   kmc.T[1, 1:jbed, :] .= Tb; 
                                   kmc.T[end, 1:jbed, :] .= Tb; 
                                  )
    bc!(kmc)

    t_series = [];
    T_series = [];
    α_series = [];

    bc!(kmc)
    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.gas) / N_active_sites)

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, pht)
        bc!(kmc)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(α_series, sum(kmc.gas) / N_active_sites)
        end
        if (iter % iterplot == 0)
            p = heatmap(kmc.T[:, :, 1])
            title!("Temperature")
            savefig(figname*"_temp-$iter.pdf")
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            p = heatmap(kmc.gas[:, :, 1])
            title!("Gas")
            savefig(figname*"_gas-$iter.pdf")
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
        end
        if (time() - last_update > 15)
            last_update = time()
            t = kmc.t
            Tavg = sum(kmc.T) / length(kmc.T)
            α = sum(kmc.gas) / N_active_sites
            @show iter, t, α, Tavg, pht.τc, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.gas) / N_active_sites)

    if doplot
        kD0 = A*exp(-EA/T0)
        kDb = A*exp(-EA/Tb)
        Cb = α_series[end]*exp(kDb*t_series[end])
        p = plot(t_series, α_series; label="kMC")
        plot!(t_series, map(t -> exp(-kD0*t), t_series); label="const. \$T = 100K\$")
        plot!(t_series, map(t -> Cb*exp(-kDb*t), t_series); label="const. \$T = 110K\$")
        xlabel!("time")
        ylabel!("gas %")
        ylims!(0.0, 1.0)
        savefig(figname*"_gas.pdf")
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        p = plot(t_series, T_series)
        xlabel!("time")
        ylabel!("avg T")
        savefig(figname*"_temp.pdf")
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        writedlm(figname*"_data.csv", hcat(t_series, α_series, T_series))
    end

    kmc, pht
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

println("My implementation of the heated desorption of a lattice gas; section IV");
println("===================================================================");
#@time main1(; doplot=true)
@time main2(; doplot=true, a=pargs["diff"], figname=pargs["figname"], 
            ΔT=pargs["dT"], maxiter=pargs["maxiter"], maxtime=pargs["maxtime"])
