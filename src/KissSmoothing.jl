module KissSmoothing
import FFTW
using Statistics

function denoise(V::Array{Float64,N}; factor::Float64=1.0, rtol::Float64=1e-12,dims::Int64=N) where N
    K1 = sqrt(2/pi)
    K2 = sqrt(2)*K1
    #K3 = sqrt(6)*K1
    #K4 = sqrt(20)*K1
    lV = size(V,dims)
    if lV<4
        return identity.(V), V.*0
    end
    iV = FFTW.dct(V)
    stri = map(i->ifelse(i==dims,lV,1),1:ndims(V))
    X = reshape(abs2.(((1:lV).-1)./(lV-1)),stri...)
    d = factor*mean(abs,diff(V,dims=dims))*(K1/K2)
    σt = 0.5
    σd = 0.25
    f = V.*0
    for iter = 1:200
        σ = sqrt(lV)*σt/(1-σt)
        if !isfinite(σ)
            break
        end
        f .= FFTW.idct(iV .* exp.(-X .* σ))
        c = mapreduce((x,y)->abs(x-y),+,f,V)/length(V)
        Δ=1-c/d
        σt+=σd*sign(Δ)
        σd/=2
        println(iter,"  ",σ,"  ",Δ)
        if abs(Δ)<rtol
            break
        end
    end
    f,V .- f
end

export denoise
end # module
