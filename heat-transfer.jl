using Distributed
@everywhere using StaticArrays
@everywhere import Base.CartesianIndex
using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf

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

function heat_transfer_events(kmc::KineticMonteCarlo, tbt::ThermalBitTrans;
                              active::Array{Bool, 3} = fill(true, size(kmc.T)))
    events = []
    c = tbt.a^2 / (kmc.h^2 * tbt.δT)
    ni, nj, nk = size(kmc.T)
    idx_ranges = chunk_idx_ranges(ni*nj*nk)
    
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

function transfer_heat_3D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
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

function transfer_heat_2D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
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

function transfer_heat_1D!(kmc::KineticMonteCarlo, pht::PoissonTrans)
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

function transfer_heat!(kmc::KineticMonteCarlo, pht::PoissonTrans)
    ni, nj, nk = size(kmc.T)

    if (minimum(size(kmc.T)) > 2)
        transfer_heat_3D!(kmc, pht)
    elseif (size(kmc.T, 2) > 2) 
        transfer_heat_2D!(kmc, pht)
    else
        transfer_heat_1D!(kmc, pht)
    end
    
    rate = 1 / pht.τc
    Δt = -log(rand()) / rate
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

function main3(; doplot::Bool=true)
    A = 1.0   # cm^2
    ℓ = 5.0   # cm
    a = 1.0   # cm/s^(1/2)
    h = 0.025  # cm
    τc = 5e-5
    xs = 0:h:ℓ
    ni = length(xs)
    T0 = 100.0
    Tb = 200.0
    analyt_T(x, t) = (Tb-T0)*erfc(x / (2*a*sqrt(t))) + T0

    @show pht = PoissonTrans(τc, a)
    @show kmc = KineticMonteCarlo(Point(ℓ, 0.0, 0.0), h; T0=T0)
    @show size(kmc.T)

    bc!(kmc::KineticMonteCarlo) = (kmc.T[1, 1, 1] = Tb)
    bc!(kmc)
    
    idx = 1
    t_outs = [0.3; 0.7; 1.3; 2.0]
    iter = 0
    p = (doplot) ? plot() : nothing
    while (kmc.t <= 1.5)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, pht)
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

    kmc, pht
end

function main4(; maxiter=Inf, doplot::Bool=true)
    th = 1.0   # cm
    ℓ = 5.0    # cm
    a = 1.0    # cm/s^(1/2)
    h = 0.2    # cm
    τc = 1e-5
    xs = 0:h:ℓ
    ni = length(xs)
    nj = ni
    @show Xs = repeat(xs, 1, nj)
    @show Ys = repeat(reshape(xs, 1, nj), ni, 1)
    T0 = 200.0
    Tb = 100.0
    analyt_T(x, y, t) = (T0-Tb)*erf(x / (2*a*sqrt(t)))*erf(y / (2*a*sqrt(t))) + Tb

    @show pht = PoissonTrans(τc, a)
    @show kmc = KineticMonteCarlo(Point(ℓ, ℓ, 0.0), h; T0=T0)
    @show size(kmc.T)

    bc!(kmc::KineticMonteCarlo) = (kmc.T[:, 1, 1] .= Tb; kmc.T[1, :, 1] .= Tb;)
    bc!(kmc)
    @show kmc.T

    idx = 1
    t_outs = [0.3; 0.8; Inf]
    iter = 0
    last_updated = time()
    while (kmc.t <= 0.9 && iter <= maxiter)
        iter += 1
        #@show iter, t_outs[idx], kmc.t, kmc.T
        transfer_heat!(kmc, pht)
        bc!(kmc)
        if (kmc.t >= t_outs[idx])
            Tavg = sum(kmc.T) / length(kmc.T)
            @show iter, t_outs[idx], kmc.t, Tavg
            if doplot
                p = scatter(vec(Xs[2:end-1,2:end-1]), vec(Ys[2:end-1,2:end-1]), vec(kmc.T[2:end-1,2:end-1]); label="\$t = $(t_outs[idx])\$, kMC")
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

if false # profile case 2
    Profile.clear()
    @time main2(; maxiter=10, doplot=false)
    @profile main2(; maxiter=1000, doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
end

if false # profile case 3
    Profile.clear()
    @time main3(; maxiter=10, doplot=false)
    @profile main3(; maxiter=1000, doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
end

if false # profile case 4
    Profile.clear()
    @time main4(; maxiter=10, doplot=false)
    @profile main4(; maxiter=1000, doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
end

println("My implementation of the pure heat transfer models in section III.A");
println("===================================================================");

println("1D thermal bit transfer...");
@time main1(; doplot=true)
println("2D thermal bit transfer...");
@time main2(; doplot=true)
println("1D Poisson heat transfer...");
@time main3(; doplot=false)
println("2D Poisson heat transfer...");
@time main4(; doplot=false)
