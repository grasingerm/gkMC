using OrdinaryDiffEq

f(u,p,t) = 1.01*u
u0 = 1/2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
using Plots
p = plot(sol; lw=5, xaxis="\$t\$", yaxis="\$u(t)\$", label="numerical")
plot!(sol.t, t->0.5*exp(1.01*t); lw=3, ls=:dash, label="exact")
println("Enter to continue")
display(p)
readline()
