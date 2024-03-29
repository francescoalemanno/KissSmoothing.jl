"""
    KissSmoothing: Easily smooth your data!

exports a single function `denoise`, for further help look at it's docstring.
"""
module KissSmoothing
using FFTW: dct, idct
using Statistics: mean
using LinearAlgebra: factorize, I
using SparseArrays: sparse

"""
    denoise(V::Array; factor=1.0, rtol=1e-12, dims=ndims(V), verbose = false)

smooth data in `V` according to:

    `V` : data to smooth

    `factor` : smoothing intensity

    `rtol` : relative tolerance on how precise the smoothing intensity is determined

    `dims` : array dimension being smoothed along

    `verbose` : enables some printing of internal info

returns a tuple (S,N) where:

    `S` : is the smoothed data

    `N` : is the extracted noise

in particular `S + N` reconstructs the original data `V`.
"""
function denoise(
    V::AbstractArray{Float64,N};
    factor::Float64 = 1.0,
    rtol::Float64 = 1e-12,
    dims::Int64 = N,
    verbose::Bool = false,
) where {N}
    buf = IOBuffer()
    K1 = sqrt(2 / pi)
    K2 = sqrt(2) * K1
    #K3 = sqrt(6)*K1
    #K4 = sqrt(20)*K1
    lV = size(V, dims)
    if lV < 4
        return copy(V), zero(V)
    end
    iV = dct(V)
    stri = map(i -> ifelse(i == dims, lV, 1), 1:ndims(V))
    X = map(abs2, reshape(LinRange(0.0, 1.0, lV), stri...))
    d = factor * mean(abs, diff(V, dims = dims)) * (K1 / K2)
    σt = 0.5
    σd = 0.25
    f = zero(V)
    for iter = 1:60
        σ = sqrt(lV) * σt / (1 - σt)
        if !isfinite(σ)
            break
        end
        f .= idct(iV .* exp.(-X .* σ))
        c = mapreduce((x, y) -> abs(x - y), +, f, V) / length(V)
        Δ = d - c
        σt += σd * sign(Δ)
        σd /= 2
        if verbose
            println(buf, iter, "  ", σ, "  ", Δ)
        end
        if abs(d - c) < rtol * d
            break
        end
    end
    if verbose
        print(String(take!(buf)))
    end
    f, V .- f
end

function tps(x::AbstractArray{Float64},y::AbstractArray{Float64})
    ϵ = nextfloat(0.0)
    r = ϵ+mapreduce(+, x, y) do a, b
        abs2(a - b)
    end
    r * log(r)
end

struct RBF{G<:AbstractArray{Float64},C<:AbstractArray{Float64}}
    Γ::G
    C::C
end

function evalPhi(xs::AbstractArray{Float64}, cp::AbstractArray{Float64})
    Phi = zeros(size(xs, 1), size(cp, 1)+1+size(xs,2))
    @views for i = 1:size(xs, 1)
        mu = 0.0
        for j = 1:size(cp, 1)
            k = tps(xs[i, :], cp[j, :])
            Phi[i, j] = k
            mu += k
        end
        mu /= size(cp, 1)
        for j = 1:size(cp, 1)
            Phi[i, j] -= mu
        end
        Phi[i, size(cp, 1)+1] = 1.0
        for j = 1:size(xs, 2)
            Phi[i, size(cp, 1)+1+j] = xs[i,j]
        end
    end
    Phi
end

function (net::RBF)(X::AbstractArray{Float64})
    evalPhi(X, net.C) * net.Γ
end


"""
    fit_rbf(xv::Array, yv::Array, cp::Array)

fit thin-plate radial basis function according to:

    `xv` : array NxP, N number of training points, P number of input variables

    `yv` : array NxQ, N number of training points, Q number of output variables

    `cp` : array KxP, K number of control points, P number of input variables

returns a callable RBF object.
"""
function fit_rbf(
    xv::AbstractArray{Float64},
    yv::AbstractArray{Float64},
    cp::AbstractArray{Float64},
)
    RBF(evalPhi(xv, cp) \ yv, collect(cp))
end
    
    
    

function basis_d(x, n1, nK)
    x1p = max(x - n1, zero(x))
    x2p = max(x - nK, zero(x))
    return ((x1p)^3 - (x2p)^3) / (nK - n1)
end

function basis_N(x, xi, k::Int)
    K = length(xi)
    if k<1 || k>K
        error("order must be between 1 and K = length(xi)")
    end
    if k == K - 1
        return one(x)
    end
    sx = (x - xi[1]) / (xi[end] - xi[1])
    if k == K
        return sx
    end
    nxi_k = (xi[k] - xi[1]) / (xi[end] - xi[1])
    nxi_em1 = (xi[end-1] - xi[1]) / (xi[end] - xi[1])
    return basis_d(sx, nxi_k, 1) - basis_d(sx, nxi_em1, xi[end])
end

"""
    fit_nspline(xv::Vector, yv::Vector, cp::Vector)

fit natural cubic splines basis function according to:

    `xv` : array N, N number of training points

    `yv` : array N, N number of training points

    `cp` : array K, K number of control points

returns a callable function.
"""
function fit_nspline(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    xi::AbstractVector{Float64},
)
    issorted(xi) || error("Knots \"xi\" must be sorted.")
    N = length(x)
    K = length(xi)
    M = zeros(N, K)
    scal = 1 / sqrt(N)
    for i = 1:N, j = 1:K
        M[i, j] = basis_N(x[i], xi, j) / scal
    end
    C = M \ (y ./ scal)
    function fn(x)
        s = zero(x)
        for i in eachindex(C)
            s += basis_N(x, xi, i) * C[i]
        end
        return s
    end
end

"""
    fit_sine_series(X::Vector,Y::Vector, basis_elements::Integer; lambda = 0.0, order=3)

fit Y ~ 1 + X + Σ sin(.) by minimising Σ (Y - f(x))^2 + lambda * ∫(Δ^order F)^2

    `X` : array N, N number of training points.

    `Y` : array N, N number of training points.

    `basis_elements` : number of sine terms.
    
    Keyword arguments:
    
    `lambda` : intensity of regularization.
    
    `order` : derivative to regularise.

returns a callable function.
"""
function fit_sine_series(X::AbstractVector{<:Real},Y::AbstractVector{<:Real}, basis_elements::Integer; lambda = 0.0, order::Int64=3)
    lx, hx = extrema(X)
    T = @. (X - lx)/(hx-lx)*pi
    M = zeros(length(X),2+basis_elements)
    for i in eachindex(X)
        M[i, 1] = 1
        M[i, 2] = T[i]
        for k in 1:basis_elements
            M[i, 2+k] = sin(k*T[i])
        end
    end
    phi = M'M
    for i=3:2+basis_elements
        phi[i,i] += lambda*(i-2)^(2order)
    end
    C = phi \ (M'*Y)
    return function fn(x)
        t = (x - lx)/(hx-lx)*pi
        s = C[1] + C[2]*t
        for k in 1:basis_elements
            s += C[2+k]*sin(k*t)
        end
        s
    end
end


"""
    lsq_denoise(S::AbstractVector{<:Real}; order::Integer=3, strength::Real = NaN)

denoise a sequence S by penalising its order-th finite differences in a least squares regression

    `S` : sequence.

    Keyword arguments:
    
    `order` : finite differencing order.
    
    `strength` : intensity of penalisation on the derivative in [0, Inf[, if unspecified it is auto-determined.

return the filtered sequence.
"""
function lsq_denoise(S::AbstractVector{<:Real}; order::Integer=3, strength::Real = NaN)
    D = sparse(1.0I, length(S), length(S))
    for _ in 1:order
        D = diff(D, dims=1)
    end
    C = D'D
    smoother(t) = factorize(I + (t/(1-t))*C)\S
    if isfinite(strength)
        return smoother(strength/(1+strength))
    end

    X = collect(S)
    for _ in 1:order
        X = diff(X)
    end
    noise = sum(x -> x^2/length(X), X) / binomial(2*order,order)

    t = 0.5
    dt = 0.25
    
    while true
        V = smoother(t)
        pnoise = mapreduce((x,y) -> abs2(x-y)/length(V), +, V, S)
        t += dt*sign(noise-pnoise)
        dt /= 2
        if abs(pnoise-noise) < 1e-5 * noise || dt < 1e-16
            break
        end
    end

    smoother(t)
end

export denoise, fit_rbf, RBF, fit_nspline, fit_sine_series, lsq_denoise
end # module
