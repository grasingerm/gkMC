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
using LinearAlgebra

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel function diffusion2D_step!(dT, T, k, Cρ, _dx, _dy)
    @inn(dT) = ( (@d_xi(k)*@d_xi(T) + @inn(k)*@d2_xi(T))*_dx^2 + 
                 (@d_yi(k)*@d_yi(T) + @inn(k)*@d2_yi(T))*_dy^2   ) / Cρ
    return
end

s = ArgParseSettings();
@add_arg_table! s begin
  "--T0"
    help = "Temperature of material as printed (K)"
    arg_type = Real 
    default = 653.0
  "--Tbed"
    help = "Bed temperature (K)"
    arg_type = Real
    default = 400.0
  "--Tair"
    help = "Temperature of material as printed (K)"
    arg_type = Real 
    default = 300.0
  "--Tc"
    help = "Crystallizaiton temperature of PEEK (K); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Real
    default = 616.0
  "--EA"
    help = "Activation energy (J / mol); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Real 
    default = 1316.0
  "--J"
    help = "Interaction potential in activation energy (J / mol) between neighbors"
    arg_type = Real 
    default = 50.0
  "--KA"
    help = "Crystallization rate constant; default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Real 
    default = 0.91
  "--k0"
    help = "Thermal conductivity of PEEK (W / m K)"
    arg_type = Real 
    default = 0.25
  "--kbed"
    help = "Thermal conductivity of copper bed (W / m K)"
    arg_type = Real
    default = 400.0
  "--kair"
    help = "Thermal conductivity of air (W / m K)"
    arg_type = Real 
    default = 5e-2
  "--Cp0"
    help = "Thermal density of PEEK (J / m^3 K)"
    arg_type = Real 
    default = 1320*2000
  "--Cpbed"
    help = "Thermal density of copper bed (J / m^3 K)"
    arg_type = Real
    default = 385*9000
  "--Cpair"
    help = "Thermal density of air (J / m^3 K)"
    arg_type = Real 
    default = 1.293*1000
  "--dT", "-d"
    help = "change in temperature due to crystallization"
    arg_type = Real
    default = 0.0
  "--ni"
    help = "lattice sites in the x-direction"
    arg_type = Int
    default = 301
  "--nj"
    help = "lattice sites in the y-direction"
    arg_type = Int
    default = 141
  "--lx"
    help = "length in the x-direction"
    arg_type = Real
    default = 0.3
  "--ly"
    help = "length in the y-direction"
    arg_type = Real
    default = 0.14
  "--tbed"
    help = "thickness of bed (m)"
    arg_type = Real
    default = 0.01
  "--tair"
    help = "air layer thickness"
    arg_type = Real
    default = 0.03
  "--l0"
    help = "initial filament"
    arg_type = Real
    default = 0.03
  "--v0"
    help = "printer head velocity"
    arg_type = Real
    default = 301.0 / (1e4)
  "--max-dt"
    help = "maximum time step for heat transfer time integration"
    arg_type = Real
    default = 1e-3
  "--tint-algo"
    help = "ODE solver for time integration (ROCK4|ROCK8|CVODE_BDF(linear_solver=:GMRES))"
    arg_type = String
    default = "ROCK4"
  "--tint-reltol"
    help = "relative tolerance for time integration"
    arg_type = Real
  "--tint-abstol"
    help = "absolute tolerance for time integration"
    arg_type = Real
  "--maxiter", "-m"
    help = "maximum number of iterations"
    arg_type = Int
    default = convert(Int, 1e7)
  "--maxtime", "-t"
    help = "maximum simulation time"
    arg_type = Real
    default = 2e4
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

function init_solver(pargs)
    solver_type = try
        eval(Meta.parse(pargs["tint-algo"]))
    catch e
        @error("Solver algorithm \"$(pargs["tint-algo"])\" not understood");
    end
    opts = Dict()
    if haskey(pargs, "tint-reltol")
        opts[:reltol] = pargs["tint-reltol"]
    end
    if haskey(pargs, "tint-abstol")
        opts[:abstol] = pargs["tint-abstol"]
    end
    return (prob) -> begin
        solve(prob, solver_type(); opts...)
    end
end

mutable struct KineticMonteCarlo
    t::Float64
    dt::Float64
    max_dt::Float64
    χ::Array{Int, 2}
    active::Array{Bool, 2}
    A::Float64
    EA::Float64
    Tc::Float64
    ΔT::Float64
    T
    Cρ
    k
    dx::Float64
    ℓx::Float64
    dy::Float64
    ℓy::Float64
    ihead::Int
    i0::Int
    T0::Float64
    v0::Float64
    jbed::Int
    jair::Int
    Cρ0::Float64
    k0::Float64
    J::Float64
    pvecs::Matrix
    solver::Function
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real,
                           jbed::Int, jair::Int, dt::Real, max_dt::Real, 
                           A::Real, EA::Real, Tc::Real, ΔT::Real,
                           Cbed::Real, C0::Real, Cair::Real,
                           kbed::Real, k0::Real, kair::Real,
                           Tbed::Real, T0::Real, Tair::Real, i0::Int, v0::Real,
                           J::Real, ndirs::Int, solver::Function)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    χ = fill(0, ni, nj)
    active = fill(false, ni, nj)
    active[1:i0, jbed+1:end-jair] .= true
    T = @zeros(ni, nj)
    T[:, 1:jbed] .= Tbed
    T[:, (jbed+1):end] .= Tair
    T[1:i0, (jbed+1):(end-jair)] .= T0
    Cρ = @zeros(ni, nj)
    Cρ[:, 1:jbed] .= Cbed
    Cρ[:, (jbed+1):end] .= Cair
    Cρ[1:i0, (jbed+1):(end-jair)] .= C0
    k = @zeros(ni, nj)
    k[:, 1:jbed] .= kbed
    k[:, (jbed+1):end] .= kair
    k[1:i0, (jbed+1):(end-jair)] .= k0
    pvecs = zeros(2, ndirs+1)
    for i in 2:(ndirs+1)
        pvecs[i, 1] = cos( 2*pi * (i-2) / ndirs )
        pvecs[i, 2] = sin( 2*pi * (i-2) / ndirs )
    end
    KineticMonteCarlo(0.0, dt, max_dt, χ, active, A, EA, Tc, ΔT, T, Cρ, k, 
                      dx, ℓx, dy, ℓy, i0, i0, T0, v0, jbed, jair, C0, k0, J,
                      pvecs, solver
                     )
end

function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    nintsteps = floor(Int, kmc.dt / kmc.max_dt)
    Δt_remaining = kmc.dt - nintsteps*kmc.max_dt
    lambda_int(Δt) = begin
        prob = ODEProblem((dT, T, p, t) -> begin
            @parallel diffusion2D_step!(dT, kmc.T, kmc.k, kmc.Cρ, 1/kmc.dx, 1/kmc.dy)
            bc!(kmc.T)
        end, kmc.T, (0.0, Δt))
        kmc.solver(prob)
    end
    for i=1:nintsteps
        sol = lambda_int(kmc.max_dt)
        kmc.T = sol[end]
        bc!(kmc.T)
    end
    sol = lambda_int(Δt_remaining)
    kmc.T = sol[end]
    bc!(kmc.T)
end

function num_crystal_nbrs(kmc, i, j, ni, nj)
    return (
             ((i > 1) ? dot(kmc.pvecs[kmc.χ[i-1, j]], kmc.pvecs[kmc.χ[i, j]]) : 0) +
             ((i < ni) ? dot(kmc.pvecs[kmc.χ[i+1, j]], kmc.pvecs[kmc.χ[i, j]]) : 0) +
             ((j > 1) ? dot(kmc.pvecs[kmc.χ[i, j-1]], kmc.pvecs[kmc.χ[i, j]]) : 0) +
             ((j < nj) ? dot(kmc.pvecs[kmc.χ[i, j+1]], kmc.pvecs[kmc.χ[i, j]]) : 0)
           )
end

function kmc_events(kmc::KineticMonteCarlo, bc!::Function)
    ni, nj = size(kmc.χ)
    nevents = ni*nj + 1
    event_handlers = Vector{Any}(undef, nevents)
    rates = zeros(nevents)

    for j=1:nj
        Threads.@threads for i=1:ni
            if !kmc.active[i, j]; continue; end
            idx = (j-1)*ni + i
            nnbrχ = num_crystal_nbrs(kmc, i, j, ni, nj)
            dEA = kmc.J*nnbrχ
            if !kmc.χ[i, j] && kmc.T[i, j] < kmc.Tc
                rates[idx] = kmc.A*exp(-(kmc.EA - dEA)/(kmc.Tc - kmc.T[i, j]))
                event_handlers[idx] = (crystallize!, (i, j))
            elseif kmc.χ[i, j] && kmc.T[i, j] > kmc.Tc
                rates[idx] = kmc.A*exp(-(kmc.EA + dEA)/(kmc.T[i, j] - kmc.Tc))
                event_handlers[idx] = (melt!, (i, j))
            end
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"

    total_crystal_rate = (length(rates) > 0) ? sum(rates) : 0.0
    kmc.dt = 1 / total_crystal_rate
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

function deposit!(kmc::KineticMonteCarlo, irange::UnitRange{Int})
    if irange.start > 1
        kmc.T[(irange.start-1):irange.stop, (kmc.jbed+1):(end-kmc.jair)] .= kmc.T0
    else
        kmc.T[irange, (kmc.jbed+1):(end-kmc.jair)] .= kmc.T0
    end
    kmc.Ci[irange, (kmc.jbed+1):(end-kmc.jair)] .= kmc.C0
    kmc.active[irange, (kmc.jbed+1):(end-kmc.jair)] .= true
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
    new_ihead = round(Int, kmc.v0*kmc.t) + kmc.i0
    if new_ihead <= size(kmc.T, 1) && new_ihead > kmc.ihead
        deposit!(kmc, (kmc.ihead+1):new_ihead)
        kmc.ihead = new_ihead
    end
end

@parallel_indices (i) function bc_top_neumann!(T)
    T[i, end] = T[i, end-1]
    return
end

@parallel_indices (i) function bc_horizontal_neumann!(T, dT, j, dj)
    @assert(abs(dj) == 1, "absolute j component of normal vector must be 1")
    T[i, j] = T[i, j-dj] - dT
    return
end

@parallel_indices (i) function bc_horizontal_dirichlet!(T, T0, j)
    T[i, j] = T0
    return
end

@parallel_indices (j) function bc_vertical_neumann!(T, dT, i, di)
    @assert(abs(di) == 1, "absolute i component of normal vector must be 1")
    T[i, j] = T[i-di, j] - dT
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
    ni, nj = size(T)

    # insulated at the top and sides
    @parallel (1:ni) bc_horizontal_neumann!(T, 0, nj, 1)
    @parallel (1:nj) bc_vertical_neumann!(T, 0, 1, -1)
    @parallel (1:nj) bc_vertical_neumann!(T, 0, ni, 1)
    
    # keep bed constant temperature
    @parallel (1:ni) bc_horizontal_dirichlet!(T, Tb, jbedbot)
    @parallel (1:ni) bc_horizontal_dirichlet!(T, Tb, jbedtop)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, 1)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, ni)
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
    ni = pargs["ni"]
    nj = pargs["nj"]
    tbed = pargs["tbed"]
    tair = pargs["tair"]
    l0 = pargs["l0"]
    v0 = pargs["v0"]
    maxdt = pargs["max-dt"]
    ℓx = pargs["lx"]
    ℓy = pargs["ly"]
    dx = ℓx/(ni-1)                  
    dy = ℓy/(nj-1)                  
    jbed = round(Int, tbed / dy)
    jair = round(Int, tair / dy)
    i0 = round(Int, l0 / dx)
    xs = 0:dx:ℓx
    ys = 0:dy:ℓy
    τc = 1e-0
    Cbed, C0, Cair = pargs["Cpbed"], pargs["Cp0"], pargs["Cpair"]
    kbed, k0, kair = pargs["kbed"], pargs["k0"], pargs["kair"]
    Tbed, T0, Tair = pargs["Tbed"], pargs["T0"], pargs["Tair"]
    J = pargs["J"]
    clims = (Tair, T0)

    kmc = KineticMonteCarlo(ℓx, dx, ℓy, dy, jbed, jair, τc, maxdt, A, EA, Tc, ΔT, 
                            Cbed, C0, Cair, kbed, k0, kair, 
                            Tbed, T0, Tair, i0, v0, J, init_solver(pargs))

    bc_curry!(T) = bc_p!(T, Tbed, 1, jbed, Tair, nj)

    t_series = [];
    T_series = [];
    α_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.χ) / sum(kmc.active))

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc_curry!)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(α_series, sum(kmc.χ) / sum(kmc.active))
        end
        if (iter % iterplot == 0)
            p = heatmap(permutedims(kmc.T[:, :, 1]); clims=clims)
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
            α = sum(kmc.χ) / sum(kmc.active)
            @show iter, t, α, Tavg, kmc.dt, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.χ) / sum(kmc.active))

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
