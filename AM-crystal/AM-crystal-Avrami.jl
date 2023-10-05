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
                    (@d_yi(Ci)*@d_yi(T) + @inn(Ci)*@d2_yi(T))*_dy^2)
    return
end

s = ArgParseSettings();
@add_arg_table! s begin
  "--T0"
    help = "Temperature of material as printed (K)"
    arg_type = Float64 
    default = 573.0
  "--Tbed"
    help = "Bed temperature (K)"
    arg_type = Float64
    default = 423.0
  "--Tair"
    help = "Temperature of material as printed (K)"
    arg_type = Float64 
    default = 300.0
  "--Ei"
  help = "Activation energy (J / mol); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 1316.0
  "--Ai"
    help = "constant for kinetic coefficient Ki model; default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 0.91
  "--n"
    help = "Avrami exponent; default from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64
    default = 2.10
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
    help = "lattice sites in the x-direction (μm)"
    arg_type = Int
    default = 9001
  "--nj"
    help = "lattice sites in the y-direction (μm)"
    arg_type = Int
    default = 141
  "--jbed"
    help = "lattice sites of heat bath (μm)"
    arg_type = Int
    default = 10
  "--jair"
    help = "air layer thickness (μm)"
    arg_type = Int
    default = 30
  "--i0"
    help = "initial filament (μm)"
    arg_type = Int
    default = 300
  "--v0"
    help = "printer head velocity (μm/s)"
    arg_type = Float64
    default = 3000.0 
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
    χ::Array{Float64, 2}
    active::Array{Bool, 2}
    Ki::Array{Float64, 2}
    Ai::Float64
    n::Float64
    Ei::Float64
    ΔT::Float64
    T
    Ci
    lam::Float64
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
    C0::Float64
    dχ::Array{Float64, 1}
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real,
                           jbed::Int, jair::Int, dt::Real, max_dt::Real, 
                           Ai::Real, Ei::Real, ΔT::Real, lam::Real, 
                           Cbed::Real, C0::Real, Cair::Real,
                           Tbed::Real, T0::Real, Tair::Real, i0::Int, v0::Real, n::Real)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    nevents = ni*nj + 1
    χ = fill(0.0, ni, nj)
    active = fill(false, ni, nj)
    active[1:i0, jbed+1:end-jair] .= true
    dχ = zeros(nevents)
    Ki = fill(0.0, ni, nj)
    T = @zeros(ni, nj)
    T[:, 1:jbed] .= Tbed
    T[:, (jbed+1):end] .= Tair
    T[1:i0, (jbed+1):(end-jair)] .= T0
    Ci = @zeros(ni, nj)
    Ci[:, 1:jbed] .= Cbed
    Ci[:, (jbed+1):end] .= Cair
    Ci[1:i0, (jbed+1):(end-jair)] .= C0
    KineticMonteCarlo(0.0, dt, max_dt, χ, active, Ki, Ai, n, Ei, ΔT, T, Ci, lam, 
                      dx, ℓx, dy, ℓy, i0, i0, T0, v0, jbed, jair, C0, dχ)
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
    nevents = ni*nj + 1
    event_handlers = Vector{Any}(undef, nevents)
    
    for j=1:nj
        Threads.@threads for i=1:ni
            if !kmc.active[i, j]; continue; end
            idx = (j-1)*ni + i
            kmc.Ki[i,j] = kmc.Ai * exp(-kmc.Ei/(kmc.T0 - kmc.T[i, j])) #Arrhenious model fitted for kinetic coeeficient Ki from (Bessard et al, J. Therm. Anal. Calorim. 2014)
            kmc.dχ[idx] = kmc.n * kmc.Ki[i,j] * (exp(-(kmc.Ki[i,j]*kmc.t)^kmc.n)) * (log((1/(exp(-(kmc.Ki[i,j]*kmc.t)^kmc.n)))^((kmc.n-1)/kmc.n))) #Rate law from (Bessard et al, J. Therm. Anal. Calorim. 2014)
            event_handlers[idx] = (crystallize!, (i, j))
            #elseif kmc.χ[i, j] && kmc.T[i, j] > kmc.Tc
                #rates[idx] = kmc.A*exp(-kmc.Ei/(kmc.T[i, j] - kmc.T0))
                #event_handlers[idx] = (melt!, (i, j))
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"

    total_crystal_rate = (length(kmc.dχ) > 0) ? sum(kmc.dχ) : 0.0
    kmc.dt = min(kmc.max_dt, 1 / total_crystal_rate) # variable time integration
    push!(kmc.dχ, 1/kmc.dt)
    push!(event_handlers, (transfer_heat!, (bc!,)))

    (rates=kmc.dχ, event_handlers=event_handlers)
end
function crystallize!(kmc::KineticMonteCarlo, i::Int, j::Int)
    kmc.χ[i, j] = 1 - exp(kmc.Ki[i,j]*kmc.t)^kmc.n #Avrami model [input model from (Bessard et al, J. Therm. Anal. Calorim. 2014) ultimately]
    kmc.T[i, j] += kmc.ΔT
end

#function melt!(kmc::KineticMonteCarlo, i::Int, j::Int)
    #@assert kmc.χ[i, j]
    #kmc.χ[i, j] = false
    #kmc.T[i, j] -= kmc.ΔT
#end

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
    Ai = pargs["Ai"]
    @show Ei = uconvert(Unitful.NoUnits, pargs["Ei"]*u"J / mol" / _NA / _kB / 1u"K")  # equivalent to Ei/R
    ΔT = pargs["dT"]
    lam = 1.0
    jbed = pargs["jbed"]
    jair = pargs["jair"]
    i0 = pargs["i0"]
    v0 = pargs["v0"]
    maxdt = pargs["max-dt"]
    ℓx = pargs["ni"]        # μm
    dx = 1                  # μm
    ℓy = pargs["nj"]        # μm
    dy = 1                  # μm
    dt = 1e-0               # seconds
    Cbed, C0, Cair = pargs["Cbed"], pargs["C0"], pargs["Cair"]
    Tbed, T0, Tair = pargs["Tbed"], pargs["T0"], pargs["Tair"]
    clims = (Tair, T0)
    n = pargs["n"]
    nj = pargs["nj"]
    ni = pargs["ni"]

    kmc = KineticMonteCarlo(ℓx, dx, ℓy, dy, jbed, jair, dt, maxdt, Ai, Ei, ΔT, 
                            lam, Cbed, C0, Cair, Tbed, T0, Tair, i0, v0, n)

    bc_curry!(T) = bc_p!(T, Tbed, 1, jbed, Tair, nj)

    t_series = [];
    T_series = [];
    χ_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χ_series, sum(kmc.χ) / sum(kmc.active))

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc_curry!)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(χ_series, sum(kmc.χ) / sum(kmc.active))
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
            χ = sum(kmc.χ) / sum(kmc.active)
            @show iter, t, χ, Tavg, kmc.dt, iter/maxiter, t/maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χ_series, sum(kmc.χ) / sum(kmc.active))

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
        writedlm(figname*"_data.csv", hcat(t_series, χ_series, T_series))
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