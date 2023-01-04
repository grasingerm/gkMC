using StaticArrays
import Base.CartesianIndex
import PhysicalConstants
using Unitful
using ArgParse;
using Plots; pyplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf
using DelimitedFiles
using OrdinaryDiffEq

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel function diffusion2D_step!(dT, T, Ci, lam, _dx, _dy)
    @inn(dT) = lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2);
    return
end

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

const _kB = PhysicalConstants.CODATA2018.BoltzmannConstant
const _NA = PhysicalConstants.CODATA2018.AvogadroConstant

mutable struct KineticMonteCarlo
    t::Float64
    dt::Float64
    const max_dt::Float64
    gas::Array{Bool, 2}
    const A::Float64
    const EA::Float64
    const ΔT::Float64
    T
    const Ci
    const lam::Float64
    const dx::Float64
    const ℓx::Float64
    const dy::Float64
    const ℓy::Float64
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real, 
                           dt::Real, max_dt::Real, A::Real, EA::Real,
                           ΔT::Real, Ci0::Real, lam::Real, jbed::Int,
                           T0::Real, Tbed::Real)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    gas = fill(true, ni, nj)
    gas[:, 1:jbed] .= false;
    T = @zeros(ni, nj)
    T[:, jbed+1:end] .= T0
    T[:, 1:jbed] .= Tbed
    Ci = @zeros(ni, nj)
    Ci .= Ci0
    KineticMonteCarlo(0.0, dt, max_dt, gas, A, EA, ΔT, 
                      T, Ci, lam, dx, ℓx, dy, ℓy)
end

function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    prob = ODEProblem((dT, T, p, t) -> begin
        @parallel diffusion2D_step!(dT, kmc.T, kmc.Ci, kmc.lam, 1/kmc.dx, 1/kmc.dy)
        bc!(kmc.T)
    end, kmc.T, (0.0, kmc.dt))
    sol = solve(prob, ROCK2(), save_everystep=false, save_start=false)
                #reltol=1e-8, abstol=1e-8)
    kmc.T = sol[end]
    Δt = -log(rand()) * kmc.dt
    kmc.t += Δt
end

function kmc_events(kmc::KineticMonteCarlo, bc!::Function)
    ni, nj = size(kmc.gas)
    nevents = ni*nj + 1
    event_handlers = Array{Tuple{Function, Tuple}}(undef, nevents)
    rates = zeros(nevents)

    for j=1:nj
        Threads.@threads for i=1:ni
            if kmc.gas[i, j]
                idx = (j-1)*ni + i
                rates[idx] = kmc.A*exp(-kmc.EA/kmc.T[i, j])
                event_handlers[idx] = (adsorp!, (i, j))
            end
        end
    end

    total_gas_rate = sum(rates)
    kmc.dt = min(kmc.max_dt, 1 / total_gas_rate) # variable time integration
    rates[end] = 1/kmc.dt
    event_handlers[end] = (transfer_heat!, (bc!,))

    (rates=rates, event_handlers=event_handlers)
end

function adsorp!(kmc::KineticMonteCarlo, i::Int, j::Int)
    kmc.gas[i, j] = false
    kmc.T[i, j] += kmc.ΔT
end

function do_event!(kmc::KineticMonteCarlo, bc!)
    events = kmc_events(kmc, bc!)
    rate_cumsum = cumsum(events.rates)
    choice_dec = rand()*rate_cumsum[end]
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    f, args = events.event_handlers[choice_idx]
    f(kmc, args...)
    Δt = -log(rand()) / rate_cumsum[end]
    kmc.t += Δt
end

function main2(; maxiter::Int=Int(1e7), iterout::Int=100, 
                 iterplot::Int=(2*maxiter),
                 maxtime::Real=1e5, doplot::Bool=true, showplot::Bool=false,
                 figname::String="graph",
                 ΔT::Real=0.0,
                 A::Real=10.0, 
                 a::Real=1.0)
    @show EA = uconvert(Unitful.NoUnits, 10.8*10^3*u"J / mol" / _kB / _NA / 1u"K")  # 1 / K
    ni = 1001
    nj = 151
    h = 2.0                       # nm
    ℓ = h*(ni-1)                  # nm
    d = h*(nj-1)                  # nm
    δT = 0.2                      # K
    xs = 0:h:ℓ
    ys = 0:h:d
    T0 = 100.0                    # K
    Tb = 110.0                    # K
    τc = 1e-0
    max_τc = Inf
    jbed = 50

    kmc = KineticMonteCarlo(ℓ, h, d, h, τc, max_τc, A, EA, ΔT, 
                            a, 1.0, jbed, T0, Tb)
    N_active_sites = sum(kmc.gas)

    bc!(T) = (
               T[:, end] = T[:, end-1];
               T[1, :] = T[2, :];
               T[end, :] = T[end-1, :];
               T[:, 1:jbed] .= Tb; 
              )
    bc!(kmc.T)

    t_series = [];
    T_series = [];
    α_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.gas) / N_active_sites)

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc!)
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
            @show iter, t, α, Tavg, kmc.dt, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.gas) / N_active_sites)

    if doplot
        paper_data = readdlm("Fig4_data.csv")
        kD0 = A*exp(-EA/T0)
        kDb = A*exp(-EA/Tb)
        Cb = α_series[end]*exp(kDb*t_series[end])
        p = plot(t_series, α_series; label="kMC")
        plot!(t_series, map(t -> exp(-kD0*t), t_series); label="const. \$T = 100K\$")
        plot!(t_series, map(t -> Cb*exp(-kDb*t), t_series); label="const. \$T = 110K\$")
        scatter!(paper_data[:, 1], paper_data[:, 2]; label="paper")
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

    kmc
end

if false # profile case 1
    Profile.clear()
    @time main2(; doplot=false)
    @profile main2(; doplot=false)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
    println("=================================================================")
    println("=================================================================")
    println("=================================================================")
end

println("Heated desorption of a lattice gas implemented with ParallelStencil; section IV");
println("===================================================================");
#@time main1(; doplot=true)
@time main2(; doplot=true, a=pargs["diff"], figname=pargs["figname"], 
            ΔT=pargs["dT"], maxiter=pargs["maxiter"], maxtime=pargs["maxtime"])
