using StaticArrays
import Base.CartesianIndex
import PhysicalConstants
using Unitful
using ArgParse;
using Plots; 
#pythonplot()
using LaTeXStrings
using SpecialFunctions
using Profile
using PProf
using DelimitedFiles
#using DifferentialEquations
using OrdinaryDiffEq
using ODEInterfaceDiffEq
using Distributions
using LinearAlgebra
using ColorSchemes, Colors
using Distributions
import YAML

const USE_GPU = false
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

#@parallel memopt=true function diffusion2D_step!(dT, T, k, Cρ, _dx, _dy)
@parallel function diffusion2D_step!(dT, T, k, Cρ, _dx, _dy)
    @inn(dT) = ( (@d_xi(k)*@d_xi(T) + @inn(k)*@d2_xi(T))*_dx^2 + 
                 (@d_yi(k)*@d_yi(T) + @inn(k)*@d2_yi(T))*_dy^2   ) / @inn(Cρ)
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
  "--Tg"
    help = "Glass transition temperature of PEEK (K)"
    arg_type = Float64
    default = 416.15
  "--EA"
    help = "Activation energy (J / mol); default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 1316.0
  "--M"
    help = "Rotational mobility constant"
    arg_type = Float64 
    default = 1.0
  "--J"
    help = "Interaction potential in activation energy (J / mol) between neighbors"
    arg_type = Float64 
    default = 1316.0
  "--Jm"
    help = "Interaction potential in reorientation (J / mol) between neighbors"
    arg_type = Float64 
    default = 130.0
  "--turnoff-meltint"
    help = "flag for whether melting considers neighboring structure"
    action = :store_true
  "--ndirs"
    help = "Number of discrete crystal plane directions"
    arg_type = Int 
    default = 8
  "--sigma-init"
    help = "Standard deviation of initial print state"
    arg_type = Float64 
    default = Inf
  "--KA"
    help = "Crystallization rate constant; default value from (Bessard et al, J. Therm. Anal. Calorim. 2014)"
    arg_type = Float64 
    default = 0.91
  "--k0"
    help = "Thermal conductivity of PEEK (W / cm K)"
    arg_type = Float64 
    default = 0.25e-2
  "--k0mult"
    help = "Thermal conductivity multiplier of crystalline PEEK"
    arg_type = Float64 
    default = 2
  "--kbed"
    help = "Thermal conductivity of copper bed (W / cm K)"
    arg_type = Float64
    default = 400.0e-2
  "--kair"
    help = "Thermal conductivity of air (W / cm K)"
    arg_type = Float64 
    default = 3e-4
  "--Cp0"
    help = "Thermal density of PEEK (J / cm^3 K)"
    arg_type = Float64 
    default = 1320.0*2000.0e-6
  "--Cpbed"
    help = "Thermal density of copper bed (J / cm^3 K)"
    arg_type = Float64
    default = 385.0*9000.0e-6
  "--Cpair"
    help = "Thermal density of air (J / cm^3 K)"
    arg_type = Float64 
    default = 1.293*1000.0e-6
  "--dT", "-d"
    help = "change in temperature due to crystallization"
    arg_type = Float64
    default = 50.0
  "--ni"
    help = "lattice sites in the x-direction"
    arg_type = Int
    default = 30
  "--nj"
    help = "lattice sites in the y-direction"
    arg_type = Int
    default = 30
  "--lx"
    help = "length in the x-direction (cm)"
    arg_type = Float64
    default = 0.15
  "--ly"
    help = "length in the y-direction (cm)"
    arg_type = Float64
    default = 0.15
  "--tbed"
    help = "thickness of bed (cm)"
    arg_type = Float64
    default = 0.01
  "--hrow"
    help = "thickness of a single row (cm)"
    arg_type = Float64
    default = 0.065
  "--wrow"
    help = "width of a single row (cm)"
    arg_type = Float64
    default = 0.08
  "--top-insulated"
    help = "flag for whether top is insulated or open (i.e. not insulated)"
    action = :store_true
  "--max-dt"
    help = "maximum time step for heat transfer time integration (s)"
    arg_type = Float64
    default = 1e-8
  "--max-Deltat"
    help = "maximum time step for kmc (s)"
    arg_type = Float64
    default = 2.5e-3
  "--tint-algo"
    help = "ODE solver for time integration (Heun|Ralston|RK4|RK8|ROCK4|ROCK8|ESERK4|ESERK5|RadauIIA3|RadauIIA5|radau|Tsit5|TsitPap8|MSRK5|MSRK6|Stepanov5|Alshina6|BS3|ImplicitEuler|ImplicitMidpoint|Trapezoid|SDIRK2|Kvaerno3|Cash4)\nHeun - explicit RK, 2nd order Heun's method with Euler adaptivity | \nRalston - explicit RK with 2nd order midpoint plus Euler adaptivity | \nRK4 - explicit 4th order RK | \nMSRK5 - explicit 5th order RK | \nMSRK6 - explicit 6th order RK | \nROCK2 - stabilized explicit 2nd order RK | \nROCK4 - stabilized explicit 4th order RK | \nROCK8 - stabilized explicit 8th order | \nESERK4 - stabilized explicit 4th order RK with extrapolation | \nESERK5 - stabilized explicit 5th order RK with extrapolation | \nRadauIIA3 - stable fully implicit 3rd order RK | \nRadauIIA5 - stable fully implicit 5th order RK | \nradau - implicit RK of variable order between 5 and 13 | \n Tsit5 - Tsitouras 5/4 Runge-Kutta method. (free 4th order interpolant) | \n TsitPap8 - Tsitouras-Papakostas 8/7 Runge-Kutta method | \n MSRK5 - Stepanov 5th-order Runge-Kutta method | \n MSRK6 - Stepanov 6th-order Runge-Kutta method | \n Stepanov5 - Stepanov adaptive 5th-order Runge-Kutta method | \n Alshina6 - Alshina 6th-order Runge-Kutta method | \n BS3 - Bogacki-Shampine 3/2 method | \n ImplicitEuler - 1st order implicit | \n ImplicitMidpoint - 2nd order implicit symplectic and symmetric | \n Trapezoid - 2nd order A stable, aka Crank-Nicolson | \n SDIRK2 - ABL stable 2nd order | \n Kvaerno3 - AL stable, stiffly accurate 3rd order | \n Cash4 - AL stable 4th order"
    arg_type = String
    default = "Tsit5"
  "--tint-reltol"
    help = "relative tolerance for time integration"
    arg_type = Float64
  "--tint-abstol"
    help = "absolute tolerance for time integration"
    arg_type = Float64
  "--maxiter", "-m"
    help = "maximum number of iterations"
    arg_type = Int
    default = convert(Int, 1e8)
  "--maxtime", "-t"
    help = "maximum simulation time"
    arg_type = Float64
    default = 6e1
  "--timeout"
    help = "Time per output (s)"
    arg_type = Float64
    default = 1e-1
  "--timeplot"
    help = "Time per plot (s)"
    arg_type = Float64
    default = 1e-1
  "--outdir"
    help = "out directory"
    arg_type = String
    default = "."
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
  "--plotsize"
    help = "Size of plot (px)"
    arg_type = Int
    default = 800
end

pargs = parse_args(s);

const _kB = PhysicalConstants.CODATA2018.BoltzmannConstant
const _NA = PhysicalConstants.CODATA2018.AvogadroConstant

function init_solver(pargs)
    solver_type = try
        eval(Meta.parse(pargs["tint-algo"]))
    catch e
        @error("Solver algorithm \"$(pargs["tint-algo"])\" is not understood")
    end
    opts = Dict()
    opts[:reltol] = (haskey(pargs, "tint-reltol") && pargs["tint-reltol"] != nothing) ? pargs["tint-reltol"] : 1e-3
    opts[:abstol] = (haskey(pargs, "tint-abstol") && pargs["tint-abstol"] != nothing) ? pargs["tint-abstol"] : 1e-6
    return (prob) -> solve(prob, solver_type(); save_everystep=false, save_start=false, opts...)
end

mutable struct KineticMonteCarlo
    t::Float64
    dt::Float64
    max_dt::Float64
    max_Δt::Float64
    χ::Array{Bool, 2}
    nhat::Array{Int, 2}
    active::Array{Bool, 2}
    A::Float64
    EA::Float64
    M::Float64
    Tc::Float64
    Tg::Float64
    ΔT::Float64
    T
    Cρ
    k
    k0mult::Float64
    dx::Float64
    ℓx::Float64
    dy::Float64
    ℓy::Float64
    T0::Float64
    Cρ0::Float64
    k0::Float64
    J::Float64
    Jm::Float64
    pvecs::Matrix
    solver::Function
    has_meltint::Bool
    d_init::Distribution
end

function KineticMonteCarlo(ℓx::Real, dx::Real, ℓy::Real, dy::Real,
                           jbed::Int, irow::Int, jrow::Int, dt::Real, 
                           max_dt::Real, max_Δt::Real,
                           A::Real, EA::Real, M::Real, Tc::Real, Tg::Real, ΔT::Real,
                           Cbed::Real, C0::Real, Cair::Real,
                           kbed::Real, k0::Real, kair::Real, k0mult::Real,
                           Tbed::Real, T0::Real, Tair::Real,
                           J::Real, Jm::Real, ndirs::Int, σ_init::Float64,
                           solver::Function, has_meltint::Bool)
    ni, nj = length(0:dx:ℓx), length(0:dy:ℓy)
    χ = fill(false, ni, nj)
    # randomly initialize polymer directions based on truncated Gaussian
    d = if σ_init == Inf
        Uniform(0.0, ndirs)
    else
        Truncated(Normal(ndirs / 2, σ_init), 0.0, ndirs)
    end
    #nhat = map(x -> ceil(Int, x), rand(d, ni, nj))
    nhat = rand(1:ndirs, ni, nj)
    active = fill(false, ni, nj)
    T = @zeros(ni, nj)
    T[:, :] .= Tair
    T[:, 1:jbed] .= Tbed
    Cρ = @zeros(ni, nj)
    Cρ[:, :] .= Cair
    Cρ[:, 1:jbed] .= Cbed
    k = @zeros(ni, nj)
    k[:, :] .= kair
    k[:, 1:jbed] .= kbed
    imid = round(Int, ni / 2)
    irow_start::Int = imid - floor(Int, irow / 2)
    irow_end::Int = irow_start + irow + 1
    for j=1:jrow
        ext::Int = round(Int, (jrow - 1) / 2 - abs( (j-1) - (jrow-1) / 2 ))
        irange = (irow_start-ext):(irow_end+ext)
        jidx::Int = jbed+j
        T[irange, jidx] .= T0
        k[irange, jidx] .= k0
        Cρ[irange, jidx] .= C0
        active[irange, jidx] .= true
        nhat[irange, jidx] = map(x -> ceil(Int, x), rand(d, length(irange)))
    end

    pvecs = zeros(2, ndirs)
    for i in 1:ndirs
        pvecs[1, i] = cos( 2*pi * (i) / ndirs )
        pvecs[2, i] = sin( 2*pi * (i) / ndirs )
    end
    KineticMonteCarlo(0.0, dt, max_dt, max_Δt, χ, nhat, active, 
                      A, EA, M, Tc, Tg, ΔT, T, Cρ, k, k0mult,
                      dx, ℓx, dy, ℓy, T0, C0, k0, J, Jm, pvecs, 
                      solver, has_meltint, d)
end

function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    nintsteps = floor(Int, kmc.dt / kmc.max_dt)
    Δt_remaining = kmc.dt - nintsteps*kmc.max_dt
    lambda_int(Δt) = begin
        prob = ODEProblem((dT, T, p, t) -> begin
            @parallel diffusion2D_step!(dT, kmc.T, kmc.k, kmc.Cρ, 1/kmc.dx, 1/kmc.dy)
        end, kmc.T, (0.0, Δt))
        kmc.solver(prob)
    end
    for i=1:nintsteps
        sol = lambda_int(kmc.max_dt)
        kmc.T = sol[end]
        bc!(kmc.T)
    end
    if Δt_remaining > eps()
        sol = lambda_int(Δt_remaining)
        kmc.T = sol[end]
        bc!(kmc.T)
    end
end

function status_crystal_nbrs(kmc, i, j, ni, nj)
    (
         ((i > 1) ?  ((dot(kmc.pvecs[kmc.nhat[i-1, j]], 
                                         kmc.pvecs[kmc.nhat[i, j]])+1)) : 0) +
         ((i < ni) ? ((dot(kmc.pvecs[kmc.nhat[i+1, j]], 
                                         kmc.pvecs[kmc.nhat[i, j]])+1)) : 0) +
         ((j > 1) ?  ((dot(kmc.pvecs[kmc.nhat[i, j-1]], 
                                         kmc.pvecs[kmc.nhat[i, j]])+1)) : 0) +
         ((j < nj) ? ((dot(kmc.pvecs[kmc.nhat[i, j+1]], 
                                         kmc.pvecs[kmc.nhat[i, j]])+1)) : 0)
    ) / 8.0
end

get_ndirs(kmc) = size(kmc.pvecs, 2)

function kmc_events(kmc::KineticMonteCarlo, bc!::Function)
    ni, nj = size(kmc.χ)
    nsites = ni*nj
    nevents = 2*nsites + 1
    event_handlers = Vector{Any}(undef, nevents)
    rates = zeros(nevents)

    for j=1:nj
        Threads.@threads for i=1:ni
            if !kmc.active[i, j]; continue; end
            if !kmc.χ[i, j] && kmc.T[i, j] < kmc.Tc
                idx = (j-1)*ni + i
                nbrχ = (1 - status_crystal_nbrs(kmc, i, j, ni, nj))
                dEA = kmc.J*nbrχ
                rates[idx] = kmc.A*exp(-(kmc.EA + dEA)/(kmc.Tc - kmc.T[i, j]))
                event_handlers[idx] = (crystallize!, (i, j))
                if kmc.T[i, j] > kmc.Tg
                    nhat_temp = kmc.nhat[i, j]
                    dθ = rand([-1; 1])
                    nhat = if kmc.nhat[i, j] + dθ < 1
                        kmc.nhat[i, j] += (dθ + get_ndirs(kmc))
                    elseif kmc.nhat[i, j] + dθ > get_ndirs(kmc)
                        newdir = mod(kmc.nhat[i, j] + dθ, get_ndirs(kmc))
                        kmc.nhat[i, j] = (newdir == 0) ? get_ndirs(kmc) : newdir
                    else
                        kmc.nhat[i, j] += dθ
                    end
                    old_nbrχ = 1 - nbrχ
                    new_nbrχ = status_crystal_nbrs(kmc, i, j, ni, nj)
                    kmc.nhat[i, j] = nhat_temp # reset
                    dE = -kmc.Jm * (new_nbrχ - nbrχ)
                    rates[idx+nsites] = kmc.M*exp(-dE/kmc.T[i, j] - kmc.Jm/(kmc.T[i, j] - kmc.Tg))
                    #rates[idx+nsites] = kmc.M*exp(-1/(kmc.T[i, j] - kmc.Tg))
                    event_handlers[idx+nsites] = (reorient!, (i, j, nhat))
                end
            elseif kmc.χ[i, j] && kmc.T[i, j] > kmc.Tc
                idx = (j-1)*ni + i
                dEA = if kmc.has_meltint
                    nbrχ = (1 - status_crystal_nbrs(kmc, i, j, ni, nj))
                    kmc.J*nbrχ
                else
                    0
                end
                rates[idx] = kmc.A*exp(-(kmc.EA - dEA)/(kmc.T[i, j] - kmc.Tc))
                event_handlers[idx] = (melt!, (i, j))
            end
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"

    total_other_events_rate = (length(rates) > 0) ? sum(rates) : 0.0
    kmc.dt = min(kmc.max_Δt, 1 / total_other_events_rate)
    rates[end] = 1/kmc.dt
    event_handlers[end] = (transfer_heat!, (bc!,))

    (rates=rates, event_handlers=event_handlers)
end

function reorient!(kmc::KineticMonteCarlo, i::Int, j::Int, nhat_idx::Int)
    @assert !kmc.χ[i, j]
    kmc.nhat[i, j] = nhat_idx
end

function crystallize!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert !kmc.χ[i, j]
    kmc.χ[i, j] = true
    kmc.T[i, j] += kmc.ΔT
    kmc.k[i, j] *= kmc.k0mult
end

function melt!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert kmc.χ[i, j]
    kmc.χ[i, j] = false
    kmc.nhat[i, j] = rand(1:get_ndirs(kmc))
    kmc.T[i, j] -= kmc.ΔT
    kmc.k[i, j] /= kmc.k0mult
end

function deposit!(kmc::KineticMonteCarlo, irange::UnitRange{Int}, 
                  jrange::UnitRange{Int})
    if irange.start > 1
        kmc.T[(irange.start-1):irange.stop, jrange] .= kmc.T0
    else
        kmc.T[irange, jrange] .= kmc.T0
    end
    kmc.Cρ[irange, jrange] .= kmc.Cρ0
    kmc.k[irange, jrange] .= kmc.k0
    kmc.active[irange, jrange] .= true
    ndirs = get_ndirs(kmc)
    lr::Int = (kmc.lrhead) ? 1 : 0
    kmc.nhat[irange, jrange] = map(x -> begin
                                   y = ceil(Int, x)
                                   z = mod(y - lr*ndirs / 2, ndirs)
                                   (z == 0) ? 8 : z
                               end, 
                               rand(kmc.d_init, length(irange), length(jrange)))
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
    return Δt
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

function bc_p_insulated!(T, Tb, jbedbot, jbedtop, Tair, jair)
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

function bc_p_open!(T, Tb, jbedbot, jbedtop, Tair, jair)
    ni, nj = size(T)

    # insulated at the top and sides
    @parallel (1:ni) bc_horizontal_dirichlet!(T, Tair, nj)
    @parallel (1:nj) bc_vertical_neumann!(T, 0, 1, -1)
    @parallel (1:nj) bc_vertical_neumann!(T, 0, ni, 1)
    
    # keep bed constant temperature
    @parallel (1:ni) bc_horizontal_dirichlet!(T, Tb, jbedbot)
    @parallel (1:ni) bc_horizontal_dirichlet!(T, Tb, jbedtop)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, 1)
    @parallel (1:jbedtop) bc_vertical_dirichlet!(T, Tb, ni)
end

function coarse_grain(kmc::KineticMonteCarlo)
    cgij = zeros(size(kmc.χ))
    ni, nj = size(cgij)
    for i=2:ni-1 
        Threads.@threads for j=2:nj-1 # interior
            @inbounds cgij[i,j] = if kmc.χ[i, j] == 0
                0
            else
                (
                    (kmc.χ[i-1, j]*(dot(kmc.pvecs[kmc.nhat[i-1, j]], 
                                        kmc.pvecs[kmc.nhat[i, j]])+1)) +
                    (kmc.χ[i+1, j]*(dot(kmc.pvecs[kmc.nhat[i+1, j]], 
                                        kmc.pvecs[kmc.nhat[i, j]])+1)) +
                    (kmc.χ[i, j-1]*(dot(kmc.pvecs[kmc.nhat[i, j-1]], 
                                        kmc.pvecs[kmc.nhat[i, j]])+1)) +
                    (kmc.χ[i, j+1]*(dot(kmc.pvecs[kmc.nhat[i, j+1]], 
                                        kmc.pvecs[kmc.nhat[i, j]])+1))
                ) / 8.0
            end
        end
    end

    Threads.@threads for i=2:ni-1 # halo i
        j = 1
        @inbounds cgij[i, j] = if kmc.χ[i, j] == 0
            0
        else
            (
                (kmc.χ[i-1, j]*(dot(kmc.pvecs[kmc.nhat[i-1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i+1, j]*(dot(kmc.pvecs[kmc.nhat[i+1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i, j+1]*(dot(kmc.pvecs[kmc.nhat[i, j+1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1))
            ) / 6.0
        end

        j = nj
        @inbounds cgij[i, j] = if kmc.χ[i, j] == 0
            0
        else
            (
                (kmc.χ[i-1, j]*(dot(kmc.pvecs[kmc.nhat[i-1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i+1, j]*(dot(kmc.pvecs[kmc.nhat[i+1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i, j-1]*(dot(kmc.pvecs[kmc.nhat[i, j-1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1))
            ) / 6.0
        end
    end

    Threads.@threads for j=2:nj-1 # halo j
        i = 1
        @inbounds cgij[i, j] = if kmc.χ[i, j] == 0
            0
        else
            (
                (kmc.χ[i, j-1]*(dot(kmc.pvecs[kmc.nhat[i, j-1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i, j+1]*(dot(kmc.pvecs[kmc.nhat[i, j+1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) + 
                (kmc.χ[i+1, j]*(dot(kmc.pvecs[kmc.nhat[i+1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1))
            ) / 6.0
        end

        i = ni
        @inbounds cgij[i, j] = if kmc.χ[i, j] == 0
            0
        else
            (
                (kmc.χ[i, j-1]*(dot(kmc.pvecs[kmc.nhat[i, j-1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i, j+1]*(dot(kmc.pvecs[kmc.nhat[i, j+1]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1)) +
                (kmc.χ[i-1, j]*(dot(kmc.pvecs[kmc.nhat[i-1, j]], 
                                    kmc.pvecs[kmc.nhat[i, j]])+1))
            ) / 6.0
        end
    end

    # corners
    cgij[1, 1] = kmc.χ[1, 1]*(
                    (kmc.χ[1, 2]*(dot(kmc.pvecs[kmc.nhat[1, 2]], 
                                      kmc.pvecs[kmc.nhat[1, 1]])+1)) +
                    (kmc.χ[2, 1]*(dot(kmc.pvecs[kmc.nhat[2, 1]], 
                                      kmc.pvecs[kmc.nhat[1, 1]])+1))
                ) / 4.0
    cgij[ni, 1] = kmc.χ[ni, 1]*(
                    (kmc.χ[ni, 2]*(dot(kmc.pvecs[kmc.nhat[ni, 2]], 
                                       kmc.pvecs[kmc.nhat[ni, 1]])+1)) +
                    (kmc.χ[ni-1, 1]*(dot(kmc.pvecs[kmc.nhat[ni-1, 1]], 
                                         kmc.pvecs[kmc.nhat[ni, 1]])+1))
                ) / 4.0
   cgij[1, nj] = kmc.χ[1, nj]*(
                    (kmc.χ[1, nj-1]*(dot(kmc.pvecs[kmc.nhat[1, nj-1]], 
                                         kmc.pvecs[kmc.nhat[1, nj]])+1)) +
                    (kmc.χ[2, nj]*(dot(kmc.pvecs[kmc.nhat[2, nj]], 
                                      kmc.pvecs[kmc.nhat[1, nj]])+1))
                ) / 4.0
   cgij[ni, nj] = kmc.χ[ni, nj]*(
                    (kmc.χ[ni, nj-1]*(dot(kmc.pvecs[kmc.nhat[ni, nj-1]], 
                                         kmc.pvecs[kmc.nhat[ni, nj]])+1)) +
                    (kmc.χ[ni-1, nj]*(dot(kmc.pvecs[kmc.nhat[ni-1, nj]], 
                                      kmc.pvecs[kmc.nhat[ni, nj]])+1))
                ) / 4.0

   return cgij

end

function main2(pargs)

    maxiter = pargs["maxiter"]
    timeout = pargs["timeout"]
    timeplot = pargs["timeplot"]
    maxtime = pargs["maxtime"]
    doplot = pargs["doplot"]
    showplot = pargs["showplot"]
    plot_size = pargs["plotsize"]
    outdir = pargs["outdir"]
    mkpath(outdir)
    figtype = pargs["figtype"]
    A = pargs["KA"]
    @show EA = uconvert(Unitful.NoUnits, pargs["EA"]*u"J / mol" / _NA / _kB / 1u"K")  # update this term based on material experimental data or temp relation?
    Tc = pargs["Tc"]
    Tg = pargs["Tg"]
    M = pargs["M"]
    ΔT = pargs["dT"]
    ni = pargs["ni"]
    nj = pargs["nj"]
    tbed = pargs["tbed"]
    hrow = pargs["hrow"]
    wrow = pargs["wrow"]
    maxdt = pargs["max-dt"]
    maxΔt = pargs["max-Deltat"]
    ℓx = pargs["lx"]
    ℓy = pargs["ly"]
    σ_init = pargs["sigma-init"]
    @show dx = ℓx/(ni-1)                  
    @show dy = ℓy/(nj-1)                  
    @show jbed = round(Int, tbed / dy)
    @show jrow = round(Int, hrow / dy)
    @show irow = round(Int, wrow / dx)
    xs = 0:dx:ℓx
    ys = 0:dy:ℓy
    τc = 1e-0
    Cbed, C0, Cair = pargs["Cpbed"], pargs["Cp0"], pargs["Cpair"]
    kbed, k0, kair = pargs["kbed"], pargs["k0"], pargs["kair"]
    k0mult = pargs["k0mult"]
    Tbed, T0, Tair = pargs["Tbed"], pargs["T0"], pargs["Tair"]
    @show J = uconvert(Unitful.NoUnits, pargs["J"]*u"J / mol" / _NA / _kB / 1u"K")  # update this term based on material experimental data or temp relation?
    @show Jm = uconvert(Unitful.NoUnits, pargs["Jm"]*u"J / mol" / _NA / _kB / 1u"K")  # update this term based on material experimental data or temp relation?
    @show has_meltint = !(pargs["turnoff-meltint"])
    ndirs = pargs["ndirs"]
    YAML.write_file(joinpath(outdir, "args.yml"), pargs)
    #pal = palette(ColorScheme([colorant"pink"; ColorSchemes.broc.colors]))
    pal = palette(:broc)
    #pal = :RdPu 
    #=pal = if 2 <= ndirs <= 11
        Symbol("Paired_$(ndirs+1)")
    else
        :default
    end
    =#
    clims = (Tair, T0)
    climsχ = (-ndirs-1, ndirs+1)
    plot_len, plot_width = if ℓx > ℓy
	    plot_size, round(Int, plot_size*ℓy/ℓx)
    else
	    round(Int, plot_size*ℓx/ℓy), plot_size
    end

    kmc = KineticMonteCarlo(ℓx, dx, ℓy, dy, jbed, irow, jrow, τc, maxdt, maxΔt,
                            A, EA, M, Tc, Tg, ΔT, 
                            Cbed, C0, Cair, kbed, k0, kair, k0mult,
                            Tbed, T0, Tair, J, Jm, ndirs, σ_init,
                            init_solver(pargs), has_meltint)
    @show dx, dy, dx / dy, ℓx, ℓy, ℓx / ℓy
    @show hrow, wrow, irow
 
    bc_curry!(T) = if pargs["top-insulated"]
        bc_p_insulated!(T, Tbed, 1, jbed, Tair, nj)
    else
        bc_p_open!(T, Tbed, 1, jbed, Tair, nj)
    end

    t_series = [];
    T_series = [];
    α_series = [];

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(α_series, sum(kmc.χ) / sum(kmc.active))

    time_since_out = 0.0
    time_since_plot = 0.0
    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        Δt = do_event!(kmc, bc_curry!)
        time_since_out += Δt
        time_since_plot += Δt
        if (time_since_out > timeout)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(α_series, sum(kmc.χ) / sum(kmc.active))
            time_since_out = 0.0
        end
        if (time_since_plot > timeplot)
            p = heatmap(permutedims(kmc.T[:, :]); clims=clims, size=(plot_len, plot_width))
            title!("Temperature, \$t=$(round(kmc.t; digits=1))\$")
            savefig(joinpath(outdir, "temp-$iter.$figtype"))
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            χmult = map(x -> (x == 0) ? -1 : 1, kmc.χ[:, :])
            χadd = map(x -> begin
                           if !x[2]
                               return 0
                           elseif (x[1] == 0)
                               return -1
                           else
                               return 1
                           end
                       end, zip(kmc.χ, kmc.active))
            #p = heatmap(permutedims(kmc.χ[:, :, 1] .* (kmc.nhat[:, :, 1] .- ((ndirs+1)/2)) - (kmc.χ[:, :, 1] .- 1) .* (climsχ[1]-1)); c=pal, size=(plot_len, plot_width)) #, clims=climsχ)
            p = heatmap(permutedims(kmc.active .* χmult .* (kmc.nhat + χadd)); c=pal, size=(plot_len, plot_width), clims=climsχ)
            title!("Crystallization, \$t=$(round(kmc.t; digits=1))\$")
            savefig(joinpath(outdir, "crystal-$iter.$figtype"))
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            cg = permutedims(coarse_grain(kmc))
            p = heatmap(cg, size=(plot_len, plot_width), clims=(0.0, 1.0))
            title!("CG Crystallization, \$t=$(round(kmc.t; digits=1))\$")
            savefig(joinpath(outdir, "cg-crystal-$iter.$figtype"))
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            writedlm(joinpath(outdir, "temp-$iter.csv"), permutedims(kmc.T), ',')
            writedlm(joinpath(outdir, "crystal-$iter.csv"), permutedims(kmc.active .* χmult .* (kmc.nhat + χadd)), ',')
            writedlm(joinpath(outdir, "cg-crystal-$iter.csv"), permutedims(coarse_grain(kmc)), ',')
            time_since_plot = 0.0
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
        p = plot(t_series, α_series)
        xlabel!("time")
        ylabel!("crystallizaiton %")
        ylims!(0.0, 1.0)
        savefig(joinpath(outdir, "crystal.$figtype"))
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        p = plot(t_series, T_series)
        xlabel!("time")
        ylabel!("avg T")
        savefig(joinpath(outdir, "temp.$figtype"))
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        writedlm(joinpath(outdir, "data.csv"), hcat(t_series, α_series, T_series), ',')
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
