using StaticArrays
import Base.CartesianIndex
import PhysicalConstants
using Unitful
using ArgParse;
using Plots;
#pyplot();
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
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

@parallel function diffusion2D_step!(dT, T, 𝛂i, lam, _dx, _dy)
    @inn(dT) = lam * @inn(𝛂i) * (@d2_xi(T) * _dx^2 + @d2_yi(T) * _dy^2)
    return
end

s = ArgParseSettings();
@add_arg_table! s begin
    "--thickness", "-y"
    help = "layer thickness of 3D printed part"
    arg_type = Int
    default = 100
    "--PrintVelocity", "-v"
    help = "velocity of printhead"
    arg_type = Real
    default = 3
    "--BedTemp", "-b"
    help = "Bed Temperature (K)"
    arg_type = Float64
    default = 350.0
    "--MeltTemp", "-m"
    help = "Melting Temperature of material (K)"
    arg_type = Float64
    default = 400.0
    "--PrintTemp", "-p"
    help = "Temperature of material as printed (K)"
    arg_type = Float64
    default = 450.0
    "--iterplot", "-z"
    help = "iterations per plot"
    arg_type = Int
    default = convert(Int, 1e7)
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
    crystallinity::Array{Bool,2}
    deposition::Array{Bool,2}
    const K::Float64
    const Er::Float64
    const ΔT::Float64
    T
    const 𝛂i
    const bed::Int
    const air::Int
    const lam::Float64
    ni::Int
    nj::Int
    const v::Float64
    const dx::Float64
    const Lx::Int
    const dy::Int
    const Ly::Int
    const Tm::Float64
end


function KineticMonteCarlo(ni::Real, nj::Real, Lx::Real, dx::Real, Ly::Real, dy::Real,
    dt::Real, max_dt::Real, K::Real, v::Real, Er::Real,
    ΔT::Real, 𝛂_bed::Real, 𝛂_air::Real, 𝛂_mat::Real, lam::Real, bed::Int, air::Int,
    T_mat::Real, T_bed::Real, T_air::Real, Tm::Real)
    Ni, Nj = length(0:dx:Lx), length(0:dy:Ly)
    crystallinity = fill(false, Ni, Nj)
    deposition = fill(false, Ni, Nj)
    T = @zeros(Ni, Nj)
    T[1:bed, 1:Ni] .= T_bed
    T[(bed+1):Nj, 1:Ni] .= T_air
    𝛂i = @zeros(Ni, Nj)
    𝛂i[1:bed, 1:Ni] .= 𝛂_bed
    𝛂i[(bed+1):Nj, 1:Ni] .= 𝛂_air
    KineticMonteCarlo(0.0, dt, max_dt, crystallinity, deposition, K, Er, ΔT,
        T, 𝛂i, bed, air, lam, ni, nj, v, dx, Lx, dy, Ly, Tm)
end  
    
function transfer_heat!(kmc::KineticMonteCarlo, bc!::Function)
    prob = ODEProblem((dT, T, p, t) -> begin
            @parallel diffusion2D_step!(dT, kmc.T, kmc.𝛂i, kmc.lam, 1 / kmc.dx, 1 / kmc.dy)
            bc!(kmc.T)
        end, kmc.T, (0.0, kmc.dt))
    sol = solve(prob, ROCK4(), save_everystep=false, save_start=false)
    #reltol=1e-8, abstol=1e-8)
    #sol = solve(prob, CVODE_BDF(linear_solver=:GMRES))
    kmc.T = sol[end]
    bc!(kmc.T)
end

function deposit_material!(kmc::KineticMonteCarlo, T_mat::Real, 𝛂_mat::Real, i::Int, j::Int)
    kmc.deposition[i,j] = true
    kmc.T[i,j] = T_mat
    kmc.𝛂i[i, j] = 𝛂_mat
end


function kmc_events(kmc::KineticMonteCarlo, bc!::Function)
    Ni, Nj = size(kmc.T)
    nevents = Ni * Nj + 1
    event_handlers = Array{Tuple{Function,Tuple}}(undef, nevents)
    rates = zeros(nevents)

    for i = 1:Ni 
        Threads.@threads for j = 1:Nj
            if !kmc.crystallinity[i,j] && kmc.deposition[i,j]
                idx = (i - 1) * Ni + j 
                rates[idx] = kmc.K * exp(-kmc.Er / (kmc.Tm - kmc.T[i, j]))  #
                event_handlers[idx] = (crystallize!, (i, j))
            end
        end
    end
    #@assert !(0.0 in rates[1:end-1]) "rates = $rates, findnext(x -> x == 0.0, rates, 1) = $(findnext(x -> x == 0.0, rates, 1))"

    total_growth_rate = sum(rates)
    kmc.dt = min(kmc.max_dt, 1 / total_growth_rate) # 
    rates[end] = 1 / kmc.dt
    event_handlers[end] = (transfer_heat!, (bc!,))

    (rates=rates, event_handlers=event_handlers)
end

function crystallize!(kmc::KineticMonteCarlo, i::Int, j::Int)
    @assert !kmc.crystallinity[i, j]
    kmc.crystallinity[i, j] = true
    kmc.T[i, j] += kmc.ΔT
end

function do_event!(kmc::KineticMonteCarlo, bc!, T_mat, α_mat)  #
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

    dx = round(Int, kmc.dt * kmc.v)
    for j = (kmc.air+1):(kmc.air+kmc.nj), i = (kmc.bed+kmc.ni):(kmc.bed+kmc.ni+dx)
        deposit_material!(kmc, T_mat, α_mat, i, j)
    end 
    @show kmc.ni = kmc.ni + dx
end

@parallel_indices (ix, iy) function bc_bed!(T, T_bed, bed)
    T[1, iy] = T_bed
    T[bed, iy] = T_bed
    return
end

@parallel_indices (ix, iy) function bc_ambient!(T, T_air, air) 
    T[air, iy] = T_air
    T[end, iy] = T_air
    T[ix, 1] = T_air
    T[ix, end] = T_air
    return
end

function bc_p!(T, T_bed, T_air, bed, air) #
    Ni, Nj = size(T)
    @parallel (1:Ni, 1:bed) bc_bed!(T, T_bed, bed)
    @parallel (1:air, bed:Nj) bc_ambient!(T, T_air, air)
    @parallel (1:air, (Nj-air):Nj) bc_ambient!(T, T_air, air)
    @parallel ((Ni-air):Ni, bed:Nj) bc_ambient!(T, T_air, air)
end

function main2(; maxiter::Int=Int(1e7), iterout::Int=100, iterplot::Int=(2 * maxiter), 
    maxtime::Real=1e5, doplot::Bool=true, showplot::Bool=false,
    figname::String="graph", ΔT::Real=0.0, Tm::Real=400.0, 
    T_bed::Real=350.0, T_mat::Real=450.0,
    K::Real=0.5, v::Real=5.0,  𝛂_bed::Real=1.5, 𝛂_air::Real=0.5, 
    𝛂_mat::Real=1.0, nj::Int=100, bed::Int=100, air::Int=50) 

    @show Er = uconvert(Unitful.NoUnits, 750 * u"J / mol" / _NA / _kB / 1u"K")  
    ni = 100
    Lx = 2000
    ℓx = 100
    dx = 1.0
    Ly = 2000
    dy = 1.0
    T_air = 300                    # K
    dt = 1e-0
    max_dt = 1e-0

    kmc = KineticMonteCarlo(ni, nj, Lx, dx, Ly, dy, dt, max_dt, K, v, Er, ΔT,
                             𝛂_bed, 𝛂_air, 𝛂_mat, 1.0, bed, air, T_mat,
                             T_bed, T_air, Tm)
    N_active_sites = kmc.ni * kmc.nj

    @show size(kmc.T)

    bc_curry!(T) = bc_p!(T, T_bed, T_air, bed, air)

    t_series = []
    T_series = []
    χ_series = []

    idx = 1
    iter = 0
    last_update = time()

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χ_series, sum(kmc.crystallinity) / N_active_sites)

    while (kmc.t <= maxtime && iter <= maxiter)
        iter += 1
        do_event!(kmc, bc_curry!, T_mat, 𝛂_mat)
        if (iter % iterout == 0)
            push!(t_series, kmc.t)
            push!(T_series, sum(kmc.T) / length(kmc.T))
            push!(χ_series, sum(kmc.crystallinity) / N_active_sites)
        end
        if (iter % iterplot == 0)
            p = heatmap(kmc.T[:, :, 1])
            title!("Temperature")
            #savefig(figname * "_temp-$iter.pdf")
            savefig(figname * "_temp-$iter.png")
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            Plots.closeall()
            p = heatmap(kmc.crystallinity[:, :, 1])
            title!("crystallinity")
            #savefig(figname * "_crystal-$iter.pdf")
            savefig(figname * "_crystal-$iter.png")
            if showplot
                println("Enter to quit")
                display(p)
                readline()
            end
            Plots.closeall()
        end
        if (time() - last_update > 15)
            last_update = time()
            t = kmc.t
            Tavg = sum(kmc.T) / length(kmc.T)
            χ = sum(kmc.crystallinity) / N_active_sites
            @show iter, t, χ, Tavg, kmc.dt, t / maxtime
        end
    end

    push!(t_series, kmc.t)
    push!(T_series, sum(kmc.T) / length(kmc.T))
    push!(χ_series, sum(kmc.crystallinity) / N_active_sites)

    if doplot
        p = plot(t_series, χ_series; label="kMC")
        xlabel!("time")
        ylabel!("crystallinity %")
        ylims!(0.0, 1.0)
        savefig(figname * "_crystallinity.pdf")
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        p = plot(t_series, T_series)
        xlabel!("time")
        ylabel!("avg T")
        savefig(figname * "_temp.pdf")
        if showplot
            println("Enter to quit")
            display(p)
            readline()
        end
        writedlm(figname * "_data.csv", hcat(t_series, χ_series, T_series))
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
@time main2(; doplot=true, nj=pargs["thickness"], T_mat=pargs["PrintTemp"],
    Tm=pargs["MeltTemp"], maxtime=pargs["maxtime"],
    T_bed=pargs["BedTemp"], iterplot=pargs["iterplot"], v=pargs["PrintVelocity"])



