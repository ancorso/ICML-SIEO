# This file defines a function that can be used to select the next best feature using expression optimization.

using ExprOptimization

# Get the function that will produce optimized new features
# Alg is the Expression optimization algorithm
# ind_loss is the loss function associated with an individual
# duplicate_penalty is the penalty for generating a repeated feature in an individual
# grammar is the Grammar the rulenode will come from
# symbol is the symbol used by the Grammar
function get_expr_opt_feature(alg, ind_loss, grammar, symbol = :R)
    # Generate a new feature that is optimal given the current individual
    function expr_opt_feature(individual::Array{RuleNode})
        lf = get_feature_loss(individual, ind_loss)
        optimize(alg, grammar, symbol, lf).tree
    end
end

