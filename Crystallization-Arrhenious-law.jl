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
using Distributions
using Sundials

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion2D_step!(dT, T, Ci, lam, _dx, _dy)
    @inn(dT) = lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2);
    return
end

s = ArgParseSettings();
@add_arg_table! s begin
  "--BedTemp", "-b"
    help = "Bed Temperature (K)"
    arg_type = Float64
    default = 400.0
  "--CrystallizeTemp", "-c"
    help = "Crystallizaiton Temperature of PEEK (K)"
    arg_type = Float64
    default = 423.0
  "--PrintTemp", "-p"
    help = "Temperature of material as printed (K)"
    arg_type = Float64 
    default = 653.0
  "--ni"
    help = "Y-axis grid size"
    arg_type = Int
    default = 200
  "--nj"
    help = "X-axis grid size"
    arg_type = Int
    default = 200
  "--ibed"
    help = "print bed thickness"
    arg_type = Int
    default = 25
  "--air"
    help = "air layer thickness = nj - air[input]"
    arg_type = Int
    default = 65
  "--iterplot", "-z"
    help = "iterations per plot"
    arg_type = Int
    default = convert(Int, 5e2)
  "--maxtime", "-t"
    help = "maximum simulation time"
    arg_type = Float64
    default = 1e5
end

pargs = parse_args(s);

const _kB = PhysicalConstants.CODATA2018.BoltzmannConstant
const _NA = PhysicalConstants.CODATA2018.AvogadroConstant

mutable struct KineticMonteCarlo 
    t::Float64
    dt::Float64
    const max_dt::Float64
    χc::Array{Bool, 2}
    const K::Float64
    K0::Array{Float64, 2}
    const Er::Float64 
    const ΔT::Float64
    T
    const Ci 
    const ibed::Int
    const air::Int
    const lam::Float64 
    const dx::Float64
    const ℓx::Float64
    const dy::Float64
    const ℓy::Float64
    const Tc::Float64
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real, 
                           dt::Real, max_dt::Real, K::Real, Er::Real,
                           ΔT::Real, Ci0::Real, Cair::Real, Ctp::Real, lam::Real, ibed::Int, air::Int,
                           T0::Real, Tb::Real, Tair::Real, Tc::Real) 
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    χc = fill(false, ni, nj) 
    K0 = @zeros(ni,nj)
    T = @zeros(ni, nj)
    T[(ibed+1):air, :] .= T0
    T[1:ibed, :] .= Tb 
    T[(air+1):end, :] .= Tair  
    Ci = @zeros(ni, nj)
    Ci[(1:ibed), :] .= Ci0
    Ci[(air+1):end, :] .= Cair 
    Ci[(ibed+1):(air), :] .= Ctp
    KineticMonteCarlo(0.0, dt, max_dt, χc, K, K0, Er, ΔT, 
                      T, Ci, ibed, air, lam, dx, ℓx, dy, ℓy, Tc) 
end

function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    prob = ODEProblem((dT, T, p, t) -> begin
        @parallel diffusion2D_step!(dT, kmc.T, kmc.Ci, kmc.lam, 1/kmc.dx, 1/kmc.dy)
        bc!(kmc.T)
    end, kmc.T, (0.0, kmc.dt))
    sol = solve(prob, ROCK4(), save_everystep=false, save_start=false)
                #reltol=1e-8, abstol=1e-8)
    #sol = solve(prob, CVODE_BDF(linear_solver=:GMRES))
    kmc.T = sol[end]
    bc!(kmc.T)
end

function kmc_events(kmc::KineticMonteCarlo, bc!::Function)
    ni, nj = size(kmc.χc)
    nevents = ni*nj+1 
    event_handlers = Array{Tuple{Function, Tuple}}(undef, nevents) 
    rates = zeros(nevents) 
       
    for i=(kmc.ibed+1):(kmc.air) 
        Threads.@threads for j=1:nj
            if kmc.T[i, j] > kmc.Tc 
                kmc.χc[i, j]=false
            elseif kmc.T[i, j] < kmc.Tc && !kmc.χc[i,j]
                idx = (i-1)*nj + j 
                rates[idx] = kmc.K * exp(-kmc.Er / (kmc.Tc - kmc.T[i, j]))
                event_handlers[idx] = (crystallize!, (i, j)) 
            end
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"
  

    total_growth_rate = sum(rates)
    kmc.dt = min(kmc.max_dt, 1 / total_growth_rate) # 
    rates[end] = 1/kmc.dt
    event_handlers[end] = (transfer_heat!, (bc!,))

    (rates=rates, event_handlers=event_handlers)
end

function crystallize!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert !kmc.χc[i, j]
    kmc.χc[i, j] = true
    kmc.T[i, j] += kmc.ΔT
end

function do_event!(kmc::KineticMonteCarlo, bc!)  #
    events = kmc_events(kmc, bc!)
    #println("events = "); display(events)
    rate_cumsum = cumsum(events.rates)
    #println("rate_cumsum = "); display(rate_cumsum)
    choice_dec = rand(Uniform(0, rate_cumsum[end]))
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    f, args = events.event_handlers[choice_idx]
    f(kmc, args...)
    Δt = -log(rand()) / rate_cumsum[end]
    kmc.t += Δt
    #println("T = "); display(kmc.T)
    #println("======================")
end

@parallel_indices (ix, iy) function bc_bed!(T, Tb, ibed)
    T[1, iy] = Tb
    T[20, iy] = Tb
    return
end

@parallel_indices (ix, iy) function bc_ambient!(T, Tair, air) #
    T[(air+20), iy] = Tair
    T[end, iy] = Tair
    return  
end

function bc_p!(T, Tb, Tair, ibed, air) #
    ni,nj = size(T)
    @parallel (1:ibed, 1:ni) bc_bed!(T, Tb, ibed)
    @parallel (air:ni, 1:nj) bc_ambient!(T, Tair, air) 
end

function main2(; maxiter::Int=Int(1e7), iterout::Int=100, 
                 iterplot::Int=(2*maxiter),
                 maxtime::Real=1e5, doplot::Bool=true, showplot::Bool=false,
                 figname::String="graph",
                 ΔT::Real=0.0, Tc::Real=423.0, Tb::Real=400.0, T0::Real=653.0,
                 K::Real=1.0, Ci0::Real=1.0, Cair::Real=0.00001, Ctp::Real=0.05, ni::Int=200, nj::Int=200, 
                 ibed::Int=25, air::Int=65, max_τc::Float64=1.0) # K value
    
    @show Er = uconvert(Unitful.NoUnits, 1e2*u"J / mol" / _NA / _kB / 1u"K")  # update this term based on material experimental data or temp relation?
    h = 2.0                       # nm
    ℓ = h*(ni-1)                  # nm
    d = h*(nj-1)                  # nm
    δT = 0.2                      # K
    xs = 0:h:ℓ
    ys = 0:h:d
    Tair = 300                    # K
    τc = 1e-0

    kmc = KineticMonteCarlo(ℓ, h, d, h, τc, max_τc, K, Er, ΔT, 
                            Ci0, Cair, Ctp, 1.0, ibed, air, T0, Tb, Tair, Tc) 
    N_active_sites = nj*(kmc.air - kmc.ibed) 

    bc_curry!(T) = bc_p!(T, Tb, Tair, ibed, air)

    t_series = [];
    T_series = [];
    χc_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χc_series, sum(kmc.χc) / N_active_sites)

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc_curry!)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(χc_series, sum(kmc.χc) / N_active_sites)
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
            p = heatmap(kmc.χc[:, :, 1])
            title!("crystallinity")
            savefig(figname*"_crystal-$iter.pdf")
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
            χc = sum(kmc.χc) / N_active_sites
            @show iter, t, χc, Tavg, kmc.dt, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χc_series, sum(kmc.χc) / N_active_sites)

    if doplot
        p = plot(t_series, χc_series; label="kMC")
        xlabel!("time")
        ylabel!("crystallinity %")
        ylims!(0.0, 1.0)
        savefig(figname*"_crystallinity.pdf")
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
        writedlm(figname*"_data.csv", hcat(t_series, χc_series, T_series))
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

println("Activation energy of a lattice crystal implemented with ParallelStencil; section IV");
println("===================================================================");
#@time main1(; doplot=true)
@time main2(; doplot=true, T0=pargs["PrintTemp"],
         Tc=pargs["CrystallizeTemp"], maxtime=pargs["maxtime"],
            Tb=pargs["BedTemp"], iterplot=pargs["iterplot"],
          ni=pargs["ni"], nj=pargs["nj"], ibed=pargs["ibed"], air=pargs["air"] ) 