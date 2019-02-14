include("src/utils.jl")
include("generate_data.jl")

using Serialization
using Plots
using Colors
pgfplots(size = (420,280))

dir = "/home/anthonycorso/Workspace/sci_discovery"
clibrary(:misc)
clr = :bluesreds

## Start with the advection diffusion equation
u, dx, dt = gen_adv_diff_data(u0 = (x)->0, uL=1, D = 1, v = 0.5, xrange = range(0.1,1, length=100), trange = range(0.1,1, length=200))

p_ad =plot(range(0.1,1, length=100), range(0.1,1, length=200), u, xlabel="\$x\$", ylabel="\$t\$", color = clr)

savefig(p_ad, "images/advdif.pdf")
cd(dir)



## Then do the Navier-Stokes Equations
γ = 1.4
ρ, u, v, E = get_cyl_data("../static_cyl/", 10:10, 1:256, 1:128)
p = (γ - 1)*(E .- 0.5*ρ.*(u.^2 + v.^2))

u = reshape(u, 256,128)
v = reshape(v, 256,128)
p = reshape(p, 256,128)
xr = range(-2, 10, length=256)
yr = range(-3,3, length=128)
xlab = "\$x\$"
ylab = "\$y\$"


p_u = plot(xr, yr, u, color=clr, xlabel=xlab, ylabel=ylab)
p_v = plot(xr, yr, v, color=clr, xlabel=xlab, ylabel=ylab)
p_p = plot(xr, yr, p, color=clr, xlabel=xlab, ylabel=ylab)

savefig(p_u, "images/xvel.pdf")
cd(dir)
savefig(p_v, "images/yvel.pdf")
cd(dir)
savefig(p_p, "images/pressure.pdf")
cd(dir)


# Plot the nonlinear oscillator
function mysin(x)
    if isfinite(x)
        return sin(x)
    else
        return 0
    end
end

dir1 = "pendulum_working/"
dir2 = "pendulum_two_feature/"

# Get the best individual
individual1, A1, d1 = load_ind(dir1)
individual2, A2, d2 = load_ind(dir2)
grammar = d1["grammar"]
nf = length(individual1)
dt = .001
x0 = [pi/2,0]

ddt(x) = [x[2],-sin(x[1])]
T = 22
N = length(0:dt:T)


# Allocate the data that we are going to explain
X_exact = Array{Float64}(undef, 2, N)
X_high_dim = Array{Float64}(undef, nf, N)
err_high_dim = Array{Float64}(undef, 2, N)
X_low_dim = Array{Float64}(undef, 2, N)
err_low_dim = Array{Float64}(undef, 2, N)

X_exact[:,1] = x0

symbol_dict = Dict(:x => x0)
to_data = get_individual_to_data(grammar, symbol_dict, 1)
X_high_dim[:,1] = to_data(individual1)
X_low_dim[:,1] = to_data(individual2)

for i=2:N
    X_exact[:, i] = X_exact[:, i-1] + dt*ddt(X_exact[:,i-1])
    X_high_dim[:, i] = A1' * X_high_dim[:,i-1]
    X_low_dim[:, i] = A2' * X_low_dim[:,i-1]
    err_high_dim[:,i] = (X_high_dim[[2,1], i] .- X_exact[:, i]).^2
    err_low_dim[:,i] = (X_low_dim[[2,1], i] .- X_exact[:, i]).^2
end

p_k = plot(0:dt:T, err_high_dim[1,:], label = string("Koopman Approximation - ", length(individual1), " features"), linewidth = 1, legend = :topleft, xlabel = "Iteration", ylabel="Squared Error", linecolor = RGB(3/255, 4/255, 140/255) )
plot!(0:dt:T, err_low_dim[1,:], label = string("Koopman Approximation - ", length(individual2), " features"), linewidth = 1, linestyle = :dash, linecolor = RGB(155/255, 55/255, 51/255) )

savefig(p_k, "images/pendulum.pdf")
cd(dir)

