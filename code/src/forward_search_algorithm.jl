# This file includes the forward search agorithm for finding the fittest individual
include("utils.jl")
include("expr_optimization_search.jl")

# Forward search algorithm for finding the fittest individual
# Greedily adds new features until there is no further improvement (after the specified number of retries) or the specified threshold is reached
# ind_loss_fn is the loss function for an individual
# new_feature_n is a function that generates new features
function forward_search(ind_loss_fn, new_feature_fn; write_to_disk = nothing, retries = 0, start_ind = RuleNode[], thresh = -Inf)
    println("Forward searching...")
    ind, optval, iter, retry  = copy(start_ind), Inf, 1, 0
    remove = false
    while true
        # Fine the next best individual using the "new_feature_fn"
        new_feat = new_feature_fn(ind)
        ind2 = unique([ind..., new_feat])
        remove = !(ind2 == ind)
        ind = ind2

        # Serialize the result as a save point
        (write_to_disk != nothing) && write_to_disk(ind, iter)

        # Compute the loss of this individual, print it and break if the loss is no better
        val = ind_loss_fn(ind)
        if val < thresh
            println("added feature: ", new_feat, " at iteration: ", iter, " for loss: ", val)
            println("Hit threshold. Done!")
            break
        end
        if val < optval
            println("added feature: ", new_feat, " at iteration: ", iter, " for loss: ", val)
            iter += 1
            retry = 0
            (optval = val)
        else
            println("retrying: ", retry)
            retry += 1
            if remove
                ind = ind[1:end-1]
            end
            if retry > retries
                break
            end
        end
    end
    return ind
end

# Backward search of the features that make up an individual
# The loss is computed with each feature removed and the feature whose absensce leads to teh largest improvement is removed
# the process is repeated until there is no further improvement or a threshold is hit.
function backward_search(individual, loss_fn; thresh = -Inf)
    println("Backward searching...")
    if length(individual) == 1
        return individual
    end
    loss_val = loss_fn(individual)
    while true
        new_val = []
        for i in 1:length(individual)
            push!(new_val, loss_fn([individual[1:i-1]..., individual[i+1:end]...]))
        end
        i = argmin(new_val)
        if new_val[i] < loss_val || new_val[i] < thresh
            println("removing: ", individual[i])
            loss_val = new_val[i]
            individual = [individual[1:i-1]..., individual[i+1:end]...]
        else
            break
        end
    end
    return individual
end

