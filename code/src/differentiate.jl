# utilities for differentiating the data numerically including noise reduction techniques
using LinearAlgebra
using Images

# Define global lists for caching
A_list = []
h_list = []
dim_list =[]
dA_list = []


# going off of the central difference formulas:
# df/dx = f(i + 1) - f(i-1) / 2h
# df/dx_l = -3f(i) +4 f(i+1) - f(i+2) / 2h
# df/dx_r = f(i-2) - 4 f(i-1) + 3 f[i] / 2h
function first_deriv(A, h, dim)
    f(i) = [[Colon() for j=1:dim-1]..., i, [Colon() for j=dim+1:length(size(A))]...]
    endd = size(A, dim)
    dA = zeros(size(A))
    dA[f(2:endd-1)...] = (A[f(3:endd)...] - A[f(1:endd-2)...]) / (2*h)
    dA[f(1)...] = (-3*A[f(1)...] + 4*A[f(2)...] - A[f(3)...]) / (2*h)
    dA[f(endd)...] = (3*A[f(endd)...] - 4*A[f(endd-1)...] + A[f(endd-2)...]) / (2*h)
    return dA
end

function second_deriv(A, h, dim)
    f(i) = [[Colon() for j=1:dim-1]..., i, [Colon() for j=dim+1:length(size(A))]...]
    endd = size(A, dim)
    dA = zeros(size(A))
    dA[f(2:endd-1)...] = (A[f(3:endd)...] -2*A[f(2:endd-1)...] +  A[f(1:endd-2)...]) / (h^2)
    dA[f(1)...] = (A[f(1)...] - 2*A[f(2)...] + A[f(3)...]) / (h^2)
    dA[f(endd)...] = (A[f(endd)...] - 2*A[f(endd-1)...] + A[f(endd-2)...]) / (h^2)
    return dA
end

function tv_first_deriv(A, h, dim, lambda = 1, iter = 100)
    for i = 1:length(A_list)
        if h_list[i] == h && dim_list[i] == dim && all(isapprox.(A_list[i], A))
            return copy(dA_list[i])
        end
    end
    dA = first_deriv(A, h, dim)
    dA = imROF(dA, lambda, iter)
    dA = imROF(dA, lambda, iter)
    push!(A_list, copy(A))
    push!(dA_list, copy(dA))
    push!(dim_list, dim)
    push!(h_list, h)
    return dA
end


