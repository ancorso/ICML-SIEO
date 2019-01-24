# This file defines all of the helper functions that go into processing/fitting data and defining loss functions
using ExprRules
using Statistics
using LinearAlgebra

# Load an individual form a serialized file
function load_ind(dir; model_file = "best_individual.jls", param_file = "params.jls")
    model_file =
    f = open(string(dir, model_file))
    dsol = deserialize(f)
    close(f)
    individual = copy(dsol["individual"])

    K = dsol["theta"]

    # Get the param file for the grammar
    param_stream = open(string(dir, param_file), "r")
    d = deserialize(param_stream)
    close(param_stream)
    return individual, K, d

end

# Get a function that can write and individual to disk
function get_write_to_disk(dir, params_fn; basename = "best_individual", ext = ".jls")
    function write_to_disk(individual::Array{RuleNode}, iteration)
        io = open(string(dir, basename, "_", iteration, ext), "w")
        io2 = open(string(dir, basename, ext), "w")
        d = Dict("individual" => individual, "theta" => params_fn(individual))
        serialize(io, d)
        serialize(io2, d)
        close(io)
        close(io2)
    end
end

# print the information of an individual
# including its features, loss function and associated parameters
function print_info(individual::Array{RuleNode}, grammar::Grammar, lf, params)
    syms = [get_executable(feature, grammar) for feature in individual]
    println("Data for individual: ")
    println("\tloss: ", lf(individual), "\n\texpressions: ", syms, "\n\tθ: ", params(individual))
end

# Get a function that can return the best fit parameters for an individual
# In this version, the target output is provided directly as "Y"
function get_params(grammar, symbol_dict_x::Dict, Y::AbstractArray, output_size; downsample_fn = nothing)
    ind_to_x = get_individual_to_data(grammar, symbol_dict_x, output_size)
    function params(individual::Array{RuleNode})
        X = ind_to_x(individual)
        Ycur = copy(Y)
        if downsample_fn != nothing
            X, Ycur = downsample_fn(X,Ycur)
        end
        θ = mlr(X, Ycur)
    end
end

# Get a function that can return the best fit parameters for an individual
# In this version, the target output is generated from the individual
function get_params(grammar, symbol_dict_x::Dict, symbol_dict_y::Dict, output_size)
    ind_to_x = get_individual_to_data(grammar, symbol_dict_x, output_size)
    ind_to_y = get_individual_to_data(grammar, symbol_dict_y, output_size)
    function params(individual::Array{RuleNode})
        mlr(ind_to_x(individual), ind_to_y(individual))
    end
end


# Get all possible rulenodes from a grammar up to the specified depth
function get_all_features(grammar, depth, symbol = :R)
    iter = ExpressionIterator(grammar, depth, symbol)
    collect(iter)
end

# Get all possible rulenodes from a a grammar up to the specified depth
# Then filter out duplicates based on the transformation they ultimately apply to the data
function get_all_unique_features(grammar, symbol_ref, depth, output_size, symbol = :R)
    feature_to_data = get_feature_to_data(grammar, symbol_ref)
    remove_duplicate_features(get_all_features(grammar, depth, symbol), feature_to_data)
end

# Remove features that produce the same (to machine precision) output when operating on the provided data
function remove_duplicate_features(individual::Array{RuleNode}, feature_to_data)
    X, new_individual = feature_to_data(individual[1]), [individual[1]]
    for i=2:length(features)
        println("feature: ", i)
        xnew = feature_to_data(features[i])
        skip = false
        for j=1:size(X,2)
            for k=1:size(X,1)
                if !isapprox(xnew[k], X[k,j])
                    # If they don't match on one point then don't bother checking any futher
                    break
                end
                # If we have made it to the end of the loop with no break then all of the data approximately matches so set skip to true
                (j == size(X, 2)) && (skip == true)
            end
            # if we found a match then we can skip through
            skip && break
        end
        if !skip
            push!(new_individual, features[i])
            X = hcat(X, xnew)
        end
    end
    new_individual
end

# Construct a function that can generate data from a feature provided it has
# the grammar and a symbol table mapping grammar symbols to datasets
function get_feature_to_data(grammar::Grammar, symbol_dict, output_size)
    S = SymbolTable(grammar)
    dkeys = collect(keys(symbol_dict))
    m = size(symbol_dict[dkeys[1]], 2)
    stride = convert(Int, output_size / m)
    already_seen = Dict{RuleNode, Array{Float64}}()
    # Function that generates data from a feature
    function feature_to_data(feature::RuleNode, X, i)
        if haskey(already_seen,feature)
            X[:, i] = already_seen[feature]
        else
            ex = get_executable(feature, grammar)
            for mi=1:m
                for sym in dkeys
                    S[sym] = symbol_dict[sym][:,mi]
                end
                X[stride*(mi-1) + 1: stride*mi, i] .= Core.eval(S, ex)
            end
            already_seen[feature] = X[:,i]
        end
        # Xg, Yg = meshgrid(1:xpts, 1:ypts)
        # circ = (Xg .- 43).^2 + (Yg .- 64).^2 .<= 400
        # @assert length(circ[:]) == stride
        # full = vcat([circ[:] for i=1:m]...)
        # @assert size(full, 1) == size(X,1)
        # X[full, i] .= 0
    end
end

function get_feature_to_data(grammar::Grammar, symbol_dict)
    S = SymbolTable(grammar)
    dkeys = collect(keys(symbol_dict))
    m = size(symbol_dict[dkeys[1]], 2)
    # Function that generates data from a feature
    function feature_to_data(feature::RuleNode, X, i)
        ex = get_executable(feature, grammar)
        X = []
        for i=1:m
            for sym in dkeys
                S[sym] = symbol_dict[sym][:,i]
            end
            if i==1
                X = Core.eval(S,ex)
            else
                X = hcat(X, Core.eval(S,ex))
            end
        end
        return X[:]
    end
end

# Construct a function that can create a data matrix from an individual
# The matrix will be mxn where m is the number of sample points per feature
# and n is the number of features
function get_individual_to_data(grammar::Grammar, symbol_dict, output_size)
    feature_to_data = get_feature_to_data(grammar, symbol_dict, output_size)
    # Get the data matrix associated with the individual
    function individual_to_data(individual::Array{RuleNode})
        L = length(individual)
        X = Array{Float64}(undef, output_size, L)
        for i=1:L
            feature_to_data(individual[i], X, i)
        end
        return X
    end
end

# Function to compute loss associated with omitting required features from the results
function missing_vars_loss(individual, required_features, penalty = 10)
    sum([penalty*(!in(f, individual)) for f in required_features])
end

# Compute the loss from X and Y given a regression function
function X_Y_loss(X, Y, regression_lf, individual, required_features, required_feature_penalty; downsample = nothing)
    if downsample != nothing
        X,Y = downsample(X, Y)
    end
    θ = mlr(X, Y)
    loss = regression_lf(X, θ, Y)
    if required_features != nothing
        loss += missing_vars_loss(individual, required_features, required_feature_penalty)
    end
    loss
end

# Get the loss function that can be evaluated at an individual
# regression_lf is a function that returns a difference measure between
# Y and Y_approx = X*θ
# This version uses the individual to compute the output data Y
function get_individual_loss(regression_lf, grammar::Grammar, symbol_dict_x::Dict, symbol_dict_y::Dict, output_size; required_features = nothing, required_feature_penalty = 10)
    ind_to_x = get_individual_to_data(grammar, symbol_dict_x, output_size)
    ind_to_y = get_individual_to_data(grammar, symbol_dict_y, output_size)
    function individual_loss(individual::Array{RuleNode})
        X = ind_to_x(individual)
        Y = ind_to_y(individual)
        X_Y_loss(X, Y, regression_lf, individual, required_features, required_feature_penalty)
    end
end

# Get the loss function that can be evaluated at an individual
# regression_lf is a function that returns a difference measure between
# Y and Y_approx = X*θ
# This version uses the individual to compute the output data Y
function get_individual_loss(regression_lf, grammar::Grammar, symbol_dict_x::Dict, Y::AbstractArray, output_size; required_features = nothing, required_feature_penalty = 10, downsample_fn = nothing)
    ind_to_x = get_individual_to_data(grammar, symbol_dict_x, output_size)
    function individual_loss(individual::Array{RuleNode})
        X = ind_to_x(individual)
        X_Y_loss(X, Y, regression_lf, individual, required_features, required_feature_penalty, downsample = downsample_fn)
    end
end

# Get the loss function for a single new feature
# Takes in the current individual and the loss function for the individual
# The resulting function returns the provided loss function evaluated with the new feature appended to the individual.
function get_feature_loss(individual::Array{RuleNode}, loss_fn)
    function feature_loss(feature::RuleNode, grammar::Grammar)
        loss_fn(unique([individual..., feature]))
    end
end

# Generate a random feature from the grammar (uniformly distributed over all the possibilites at the specified grammar depth)
function get_random_feature(grammar, grammar_depth, sym = :R)
    features = collect(ExpressionIterator(grammar, grammar_depth, :R))
    random_feature(individual...) = features[rand(1:length(features))]
end

# Get a random individual with the specified number of features (uses random feature generation at a desired depth)
function get_random_individual(grammar, grammar_depth, num_features, sym = :R)
    rf = get_random_feature(grammar, grammar_depth)
    random_indivdual() = [rf() for i=1:num_features]
end

# Multiple Linear regression with design matrix X and output y
function mlr(X, Y)
    try
        pinv(X'*X)*X'*Y
    catch
        zeros(size(X,2), size(Y,2)) * NaN
    end
end

# Average sum of squares loss function for generalized regression (using approximation and output)
# We divide by the number of features in Y to keep from penalizing additional features
avg_sumsq(Y_approx, Y) = sqrt(sum((Y - Y_approx).^2)) / size(Y, 2)

# Average sum of squares loss function for generalized regression (using input, params and output)
avg_sumsq(X, θ, Y) = avg_sumsq(X*θ, Y)

# normalized average sum of squares loss function (using approximation and output)
function avg_norm_sumsq(Y_approx, Y)
    μY, σY = mean(Y, dims=1), std(Y, dims=1)
    Y = (Y .- μY)./σY
    Y_approx = (Y_approx .- μY)./σY
    avg_sumsq(Y_approx, Y)
end

# normalized average sum of squares loss function (using input, params and output)
avg_norm_sumsq(X, θ, Y) = avg_norm_sumsq(X*θ, Y)

# Get the r^2 value to see how good the fit is
rsq(y_approx, y) = 1 - sum((y-y_approx).^2)/sum((y.-mean(y)).^2)
rsq(X, θ, y) = rsq(X*θ, y)

# Get the adjusted rsq value (penalizes using more features)
function adj_rsq(X, θ, Y, α = 0.1)
    n, p = size(X)
    r2 = rsq(X, θ, Y)
    r2 - α*(1- r2)*(p-1)
end

# Negative of the adj_rsq value (so it can be used as a loss function)
neg_adj_rsq(X, θ, Y, α = 0.1) = - adj_rsq(X, θ, Y, α)

# Negative of the adj_rsq value (so it can be used as a loss function)
function get_neg_adj_rsq(α)
    nadr(X,θ,Y) = neg_adj_rsq(X, θ, Y, α)
    nadr
end

# Negative of the rsq value -- so it can be used as a loss function
neg_rsq(X, θ, Y) = - rsq(X, θ, Y)

