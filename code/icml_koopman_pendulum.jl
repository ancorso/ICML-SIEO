include("src/forward_search_algorithm.jl")

# define the dynamical system under test
function gen_koopman_pendulum_data(;dt = 0.01, T = 30)
    # Define dynamics
    ddt(x) = [x[2],-sin(x[1])]

    # Allocate the data that we are going to explain
    N = length(0:dt:T)
    X = Array{Float64}(undef, 2, N)
    Y = Array{Float64}(undef, 2, N)

    theta0 = [pi/2]
    for j = 1:length(theta0)
        bgn = N*(j-1)+1
        nd = N*j
        X[:,bgn] = [theta0[j], 0.]
        for i=bgn:nd
            Y[:, i] = X[:, i] + dt*ddt(X[:, i])
            if i < nd
                X[:,i+1] = Y[:,i]
            end
        end
    end
    X, Y, dt, T
end

## Generate synthetic data
X, Y, dt, N = gen_koopman_pendulum_data(dt = 0.001, T=10)

grammar_depth = 4

mysin(x) = isfinite(x) ? sin(x) : 0

# Define the grammar
grammar = @grammar begin
    R = getindex(x, 1) | getindex(x,2) # The two variables of interest
    R = R * R # multiplication
    R = mysin(R*G + G)
    R = mysin(R*G)
    R = sqrt(abs(R))
    R = log(abs(R))
    R = R/R
    R = 1/R
    R = exp(G*R)
    R = 1/(1 + exp(-G*R))
    R = exp(-(R - G)^2/G)
    R = imag(Complex(R)^G)
    R = real(Complex(R)^G)
    G = -G
    G = G+G
    G = G*G
    G = G/G
    G = 10^H
    H =  -5.0
    H = -4.63158
    H = -4.26316
    H = -3.89474
    H = -3.52632
    H = -3.15789
    H =  -2.78947
    H =  -2.42105
    H =  -2.05263
    H =  -1.68421
    H =  -1.31579
    H =  -0.947368
    H =  -0.578947
    H =  -0.210526
    H =  0.157895
    H =  0.526316
    H =   0.894737
    H =   1.26316
    H =  1.63158
    H =   2.0
end
output_size =length(X[1,:])
symbol_dict_x = Dict(:x => X)
symbol_dict_y = Dict(:x => Y)
params = get_params(grammar, symbol_dict_x, symbol_dict_y, output_size) # used to find the associated params with an individual from the grammar

## Define the loss function (require the second state variables to be involved)
lf = get_individual_loss(avg_norm_sumsq, grammar, symbol_dict_x, symbol_dict_y, output_size, required_features = Set{RuleNode}([RuleNode(2), RuleNode(1)]), required_feature_penalty = 1e4)

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
threshold = 1e-13
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

