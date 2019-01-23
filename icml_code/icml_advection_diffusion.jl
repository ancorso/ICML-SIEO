include("src/forward_search_algorithm.jl")
include("src/differentiate.jl")
using SpecialFunctions

# Define the dynamical system under test
# The function that describes the particular adv-diff problem
function adv_diff_sol(u0, uL, D, v)
    A(x,t) = 0.5*erfc((x - v*t) / (2 * sqrt(D*t))) + 0.5*exp(v*x/D)*erfc((x + v*t) / (2 * sqrt(D*t)))
    u(x,t) = u0(x- v*t) + (uL - u0(x-v*t))*A(x,t)
end

# Get the data for the specific problem at hand including range of x and t
function gen_adv_diff_data(;u0 = (x)->0, uL = 0, D = 1, v = 1, xrange = range(0,1,length=100), trange=range(0,1,length=100))
    ufn = adv_diff_sol(u0, uL, D, v)
    up = [ufn(xi, ti) for xi in xrange, ti in trange]
    up, xrange[2] - xrange[1], trange[2] - trange[1]
end

## Generate synthetic data
x = range(0.3,1, length=100)
t = range(0.001, 0.1, length = 200)
xpts, tpts = length(x), length(t)
u, dx, dt = gen_adv_diff_data(u0 = (x)->0, uL=1, D = 1, v = 1, xrange = x, trange = t)

# define the max grammar depth
grammar_depth = 3

ddx(u) = first_deriv(reshape(u,xpts,tpts), dx, 1)[:]
ddt(u) = first_deriv(reshape(u,xpts,tpts), dt, 2)[:]
u = u[:]
y = ddt(u)[:]

## define the grammar
grammar = @grammar begin
    R = u # reference a variable
    R = ddx(R) # spatial derivative
    R = broadcast(*, R, R) # multiplication
    R = broadcast(/, R, R) # division
end
output_size = length(u)
symbol_dict = Dict(:u => u)
params = get_params(grammar, symbol_dict, y,  output_size) # used to find the associated params with an individual from the grammar

# Define the loss function
lf = get_individual_loss(get_neg_adj_rsq(0.1), grammar, symbol_dict, y, output_size)

# Grammatical Evolution
ge = GrammaticalEvolution(grammar,:R,
                          500, # pop_size
                          50, # iterations
                          grammar_depth, # init_gene_length
                          20, # max_gene_length
                          grammar_depth, # max_depth
                          0.2, # p_reproduction
                          0.4, # p_crossover
                          0.4; # p_mutation
                          select_method=GrammaticalEvolutions.TruncationSelection(
                                        8 # keep_top_k
                                        )
                          )
gen_ge_opt_feature = get_expr_opt_feature(ge, lf, grammar)

new_loss = Inf
old_loss = Inf
iteration = 1
best_individual = RuleNode[]
threshold = -.99
while iteration == 1 || new_loss < old_loss
    global best_individual = forward_search(lf, gen_ge_opt_feature, start_ind = best_individual, thresh = threshold)
    global best_individual = backward_search(best_individual, lf, thresh = threshold)
    global old_loss = new_loss
    global new_loss = lf(best_individual)
    if new_loss < threshold
        break
    end
    global iteration += 1
end

## print out the results for the individual
print_info(best_individual, grammar, lf, params)

