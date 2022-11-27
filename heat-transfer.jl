using StaticArrays

abstract type HeatTrans; end

function κ_to_a(κ::Real, σ::Real, ρ::Real) = sqrt(κ / (σ*ρ))

struct ThermalBitTrans <: HeatTrans
    a::Real
    h::Real
    δT::Real
    σρ::Real
end

struct PoissonTrans <: HeatTrans
    a::Real
end

Point = SVector{3, Float64};

struct KineticMonteCarlo
    t::Real
    h::Real
    T::Array{Float64, 3}
    botleft::Point
end

function ijk_to_pos(i::Int, j::Int, k::Int; 
                    h::Real = 1.0, botleft::Point=Point(0.0, 0.0, 0.0))
    botleft + h*Point(i-1, j-1, k-1)
end

Idx = SVector{3, Int};
CartesianIndex(idx::Idx) = CartesianIndex(Tuple(idx));
nnbrs = [
         [1; 0; 0], [-1; 0; 0],
         [0; 1; 0], [0; -1; 0],
         [0; 0; 1], [0; 0; -1]
        ];

function heat_transfer_events(kmc::KineticMonteCarlo, tbt::ThermalBitTrans;
                              active::Matrix{Bool} = fill(true, size(kmc.T)))
    events = [];
    c = tbt.a^2 / (kmc.h^2 * tbt.δt)
    ni, nj, nk = size(kmc.T)
    for i=1:ni, j=nj, k=1:nk
        J = Idx(i, j, k)
        Tj = kmc.T[CartesianIndex(J)]
        for nbr in nnbrs
            K = J + nbr
            if (0 <= K[1] <= ni && 0 <= K[2] <= nj && 0 <= K[3] <= nk && 
                active[CartesianIndex(K)])
                Tk = kmc.T[CartesianIndex(K)]
                if Tj > Tk
                    rate = c * (Tj - Tk)
                    push!(events, NamedTuple{:from, :to, :rate}(J, K, rate))
                end
            end
        end
    end
    return events
end

function transfer_heat(kmc::KineticMonteCarlo, tbt::ThermalBitTrans)
    ht_events = heat_transfer_events(kmc, tbt)
    rate_cumsum = cumsum(map(event -> event.rate, events))
    choice_dec = rand()*rate_cumsum[end]
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    J, K, rate = ht_events[choice_idx]
    kmc.T[CartesianIndex(K)] += tbt.δt
    kmc.T[CartesianIndex(J)] -= tbt.δt
    Δt = log(rand()) / sum(map(event -> event.rate, events))
    kmc.t += Δt
end
