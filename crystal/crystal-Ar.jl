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
    @inn(dT) = lam*((@d_xi(Ci)*@d_xi(T) + @inn(Ci)*@d2_xi(T))*_dx^2 + 
                    (@d_yi(Ci)*@d_yi(T) + @inn(Ci)*@d2_yi(T))*_dy^2   )
    return
end

s = ArgParseSettings();
@add_arg_table! s begin
  "--T0"
    help = "Temperature of material as printed (K)"
    arg_type = Float64 
    default = 653.0
  "--Tbed"
    help = "Bed temperature (K)"
    arg_type = Float64
    default = 400.0
  "--Tair"
    help = "Temperature of material as printed (K)"
    arg_type = Float64 
    default = 300.0
  "--Tc"
    help = "Crystallizaiton temperature of PEEK (K); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64
    default = 616.0
  "--EA"
  help = "Activation energy (J / mol); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 1316.0
  "--KA"
    help = "Crystallization rate constant; default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 0.91
  "--C0"
    help = "Thermal diffusivity of PEEK"
    arg_type = Float64 
    default = 0.25e-3
  "--Cbed"
    help = "Thermal diffusivity of copper bed"
    arg_type = Float64
    default = 0.4
  "--Cair"
    help = "Thermal diffusivity of air"
    arg_type = Float64 
    default = 5e-5
  "--dT", "-d"
    help = "change in temperature due to crystallization"
    arg_type = Float64
    default = 0.0
  "--ni"
    help = "lattice sites in the x-direction"
    arg_type = Int
    default = 301
  "--nj"
    help = "lattice sites in the y-direction"
    arg_type = Int
    default = 141
  "--jbed"
    help = "lattice sites of heat bath"
    arg_type = Int
    default = 10
  "--jair"
    help = "air layer thickness"
    arg_type = Int
    default = 30
  "--max-dt"
    help = "maximum time step"
    arg_type = Float64
    default = 1.0
  "--maxiter", "-m"
    help = "maximum number of iterations"
    arg_type = Int
    default = convert(Int, 1e7)
  "--maxtime", "-t"
    help = "maximum simulation time"
    arg_type = Float64
    default = 2e7
  "--iterout"
    help = "Iterations per output"
    arg_type = Int
    default = convert(Int, 1e3)
  "--iterplot"
    help = "Iterations per plot"
    arg_type = Int
    default = convert(Int, 1e4)
  "--figname"
    help = "figure name"
    arg_type = String
    default = "graph"
  "--figtype"
    help = "figure type"
    arg_type = String
    default = "pdf"
  "--doplot"
    help = "do plots"
    action = :store_true
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
    max_dt::Float64
    χ::Array{Bool, 2}
    active::Array{Bool, 2}
    A::Float64
    EA::Float64
    Tc::Float64
    ΔT::Float64
    T
    Ci
    lam::Float64
    dx::Float64
    ℓx::Float64
    dy::Float64
    ℓy::Float64
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real,
                           jbed::Int, jair::Int, dt::Real, max_dt::Real, 
                           A::Real, EA::Real, Tc::Real, ΔT::Real, lam::Real, 
                           Cbed::Real, C0::Real, Cair::Real,
                           Tbed::Real, T0::Real, Tair::Real)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    χ = fill(false, ni, nj)
    active = fill(false, ni, nj)
    active[:, jbed+1:end-jair] .= true
    T = @zeros(ni, nj)
    T[:, 1:jbed] .= Tbed
    T[:, (jbed+1):(end-jair-1)] .= T0
    if jair != 0
        T[:, (end-jair):end] .= Tair
    end
    Ci = @zeros(ni, nj)
    Ci[:, 1:jbed] .= Cbed
    Ci[:, jbed+1:end-jair] .= C0
    if jair != 0
        Ci[:, (end-jair):end] .= Cair
    end
    KineticMonteCarlo(0.0, dt, max_dt, χ, active,
                      A, EA, Tc, ΔT, T, Ci, lam, dx, ℓx, dy, ℓy)
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
    ni, nj = size(kmc.χ)
    nevents = sum(kmc.active)+1
    event_handlers = []
    rates = Float64[]

    for j=1:nj
        Threads.@threads for i=1:ni
            if !kmc.active[i, j]; continue; end
            if !kmc.χ[i, j] && kmc.T[i, j] < kmc.Tc
                push!(rates, kmc.A*exp(-kmc.EA/(kmc.Tc - kmc.T[i, j])))
                push!(event_handlers, (crystallize!, (i, j)))
            elseif kmc.χ[i, j] && kmc.T[i, j] > kmc.Tc
                push!(rates, kmc.A*exp(-kmc.EA/(kmc.T[i, j] - kmc.Tc)))
                push!(event_handlers, (melt!, (i, j)))
            end
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"

    total_crystal_rate = (length(rates) > 0) ? sum(rates) : 0.0
    kmc.dt = min(kmc.max_dt, 1 / total_crystal_rate) # variable time integration
    push!(rates, 1/kmc.dt)
    push!(event_handlers, (transfer_heat!, (bc!,)))

    (rates=rates, event_handlers=event_handlers)
end

function crystallize!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert !kmc.χ[i, j]
    kmc.χ[i, j] = true
    kmc.T[i, j] += kmc.ΔT
end

function melt!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert kmc.χ[i, j]
    kmc.χ[i, j] = false
    kmc.T[i, j] -= kmc.ΔT
end

function do_event!(kmc::KineticMonteCarlo, bc!)
    events = kmc_events(kmc, bc!)
    rate_cumsum = cumsum(events.rates)
    choice_dec = rand(Uniform(0, rate_cumsum[end]))
    choice_idx = (searchsorted(rate_cumsum, choice_dec)).start
    f, args = events.event_handlers[choice_idx]
    f(kmc, args...)
    Δt = -log(rand()) / rate_cumsum[end]
    kmc.t += Δt
    #println("T = "); display(kmc.T)
    #println("======================")
end

@parallel_indices (i) function bc_top_neumann!(T)
    T[i, end] = T[i, end-1]
    return
end

@parallel_indices (i) function bc_horizontal_dirichlet!(T, T0, j)
    T[i, j] = T0
    return
end

@parallel_indices (j) function bc_vertical_dirichlet!(T, T0, i)
    T[i, j] = T0
    return
end

@parallel_indices (j) function bc_periodic!(T)
    T[1, j] = T[end-1, j]
    T[end, j] = T[2, j]
    return
end

function bc_p!(T, Tb, jbedbot, jbedtop, Tair, jair)
    @parallel (1:size(T,2)) bc_periodic!(T)
    @parallel (1:size(T,1)) bc_horizontal_dirichlet!(T, Tb, jbedbot)
    @parallel (1:size(T,1)) bc_horizontal_dirichlet!(T, Tb, jbedtop)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, 1)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, size(T,1))
    #@parallel (1:size(T,1)) bc_horizontal_dirichlet!(T, Tair, jair)
    @parallel (1:size(T,1)) bc_top_neumann!(T)
end

function main2(pargs)

    maxiter = pargs["maxiter"]
    iterout = pargs["iterout"]
    iterplot = pargs["iterplot"]
    maxtime = pargs["maxtime"]
    doplot = pargs["doplot"]
    showplot = pargs["showplot"]
    figname = pargs["figname"]
    figtype = pargs["figtype"]
    A = pargs["KA"]
    @show EA = uconvert(Unitful.NoUnits, pargs["EA"]*u"J / mol" / _NA / _kB / 1u"K")  # update this term based on material experimental data or temp relation?
    Tc = pargs["Tc"]
    ΔT = pargs["dT"]
    lam = 1.0
    ni = pargs["ni"]
    nj = pargs["nj"]
    jbed = pargs["jbed"]
    jair = pargs["jair"]
    maxdt = pargs["max-dt"]
    h = 2.0                       # nm
    ℓ = h*(ni-1)                  # nm
    d = h*(nj-1)                  # nm
    xs = 0:h:ℓ
    ys = 0:h:d
    τc = 1e-0
    Cbed, C0, Cair = pargs["Cbed"], pargs["C0"], pargs["Cair"]
    Tbed, T0, Tair = pargs["Tbed"], pargs["T0"], pargs["Tair"]

    kmc = KineticMonteCarlo(ℓ, h, d, h, jbed, jair, τc, maxdt, A, EA, Tc, ΔT, 
                            lam, Cbed, C0, Cair, Tbed, T0, Tair)
    N_active_sites = sum(kmc.active)

    bc_curry!(T) = bc_p!(T, Tbed, 1, jbed, Tair, nj)

    t_series = [];
    T_series = [];
    α_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.χ) / N_active_sites)

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc_curry!)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(α_series, sum(kmc.χ) / N_active_sites)
        end
        if (iter % iterplot == 0)
            p = heatmap(permutedims(kmc.T[:, :, 1]))
            title!("Temperature")
            savefig(figname*"_temp-$iter.$figtype")
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            p = heatmap(permutedims(kmc.χ[:, :, 1]))
            title!("Crystallization")
            savefig(figname*"_crystal-$iter.$figtype")
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
            α = sum(kmc.χ) / N_active_sites
            @show iter, t, α, Tavg, kmc.dt, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.χ) / N_active_sites)

    if doplot
        p = plot(t_series, χ_series)
        xlabel!("time")
        ylabel!("crystallizaiton %")
        ylims!(0.0, 1.0)
        savefig(figname*"_crystal.$figtype")
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        p = plot(t_series, T_series)
        xlabel!("time")
        ylabel!("avg T")
        savefig(figname*"_temp.$figtype")
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
    @time main2(pargs)
    @profile main2(pargs)
    Profile.print()
    pprof()
    println("Enter to continue...")
    readline()
    println("=================================================================")
    println("=================================================================")
    println("=================================================================")
end

println("Crystallization of PEEK during AM")
println("===================================================================");
@time main2(pargs)
