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
    V::Array{Float64,N};
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
    X = reshape(abs2.(((1:lV) .- 1) ./ (lV - 1)), stri...)
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
        if abs(d-c) < rtol*d
            break
        end
    end
    if verbose
        print(String(take!(buf)))
    end
    f, V .- f
end

export denoise
end # module
