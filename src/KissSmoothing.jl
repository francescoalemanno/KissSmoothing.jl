"""
    KissSmoothing: Easily smooth your data!

exports a single function `denoise`, for further help look at it's docstring.
"""
module KissSmoothing
using FFTW: dct, idct
using Statistics: mean

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
const ϵ::Float64 = nextfloat(0.0)

function tps(x::AbstractArray{Float64},y::AbstractArray{Float64})
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
    Phi = zeros(size(xs, 1), size(cp, 1)+1)
    for i = 1:size(xs, 1)
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

export denoise, fit_rbf, RBF
end # module
