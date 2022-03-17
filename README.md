# KissSmoothing.jl

This package implements a smoothing procedure

    denoise(V::Array; factor=1.0, rtol=1e-12, dims=N, verbose = false)

smooth data in `V` according to:

    `V` : data to smooth

    `factor` : smoothing intensity

    `rtol` : relative tolerance on how precise the smoothing intensity is determined

    `dims` : array dimension being smoothed

    `verbose` : enables some printing of internal info

returns a tuple (S,N) where:

    `S` : is the smoothed data

    `N` : is the extracted noise

in particular `S + N` reconstructs the original data `V`.


# Example

```julia
using KissSmoothing, Statistics, LinearAlgebra
using PyPlot

for (i,s) in enumerate(2 .^ LinRange(-1.5,1.5,4))
    # generating a simple sinusoidal signal
    X = LinRange(0,2pi,1000)
    Y = sin.(X)
    # generate it's noise corrupted version
    TN = std(Y).*randn(length(X))./7 .*s
    raw_S = Y .+ TN
    # using this package function to extract signal S and noise N
    S, N = denoise(raw_S)

    subplot(2,2,i)
    plot(X,raw_S, color="gray",lw=0.8)
    plot(X,Y,color="red")
    plot(X,S,color="blue", ls ="dashed")
end
tight_layout()
savefig("test.png")
```
![test.png](test.png "Plot of 1D signal smoothing")

## Multidimensional example
```julia

using KissSmoothing, Statistics, LinearAlgebra
using PyPlot
figure()
for (i,s) in enumerate(2 .^ LinRange(-1.5,1.5,4))
    # generating a simple sinusoidal signal
    X = LinRange(0,10pi,1000)
    Y = sin.(X) .+ randn(length(X))./7 .*s
    Z = cos.(X) .+ randn(length(X))./7 .*s
    M = [Y Z]
    # using this package function to extract signal S and noise N
    S, N = denoise(M, dims=1)

    subplot(2,2,i)
    scatter(M[:,1],M[:,2], color="gray",s=2)
    plot(S[:,1],S[:,2], color="red",lw=1.5)
end
tight_layout()
savefig("test_multi.png")
```
![test_multi.png](test_multi.png "Plot of multidim smoothing")
