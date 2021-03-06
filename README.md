# KissSmoothing.jl
This package implements a denoising procedure, and a Radial Basis Function estimation procedure
## Denoising

    denoise(V::Array; factor=1.0, rtol=1e-12, dims=ndims(V), verbose = false)

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


### Example

```julia
using KissSmoothing, Statistics, LinearAlgebra
using PyPlot
figure(figsize=(5,4))
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
    plot(X,raw_S, color="gray",lw=0.8, label="Y noisy")
    plot(X,Y,color="red",label="Y true")
    plot(X,S,color="blue", ls ="dashed",label="Y smoothed")
    xlabel("X")
    ylabel("Y")
    i==1 && legend()
end
tight_layout()
savefig("test.png")
```
![test.png](test.png "Plot of 1D signal smoothing")

### Multidimensional example
```julia
using KissSmoothing, Statistics, LinearAlgebra
using PyPlot
figure(figsize=(5,4))
for (i,s) in enumerate(2 .^ LinRange(-1.5,1.5,4))
    # generating a simple circle dataset
    X = LinRange(0,10pi,1000)
    Y = sin.(X) .+ randn(length(X))./7 .*s
    Z = cos.(X) .+ randn(length(X))./7 .*s
    M = [Y Z]
    O = [sin.(X) cos.(X)]
    # using this package function to extract signal S and noise N
    S, N = denoise(M, dims=1)

    subplot(2,2,i)
    scatter(M[:,1],M[:,2], color="gray",s=2,label="noisy")
    plot(S[:,1],S[:,2], color="red",lw=1.5,label="smoothed")
    plot(O[:,1],O[:,2], color="blue",lw=1.0,label="true")
    i==1 && legend()
    xlabel("X")
    ylabel("Y")
end
tight_layout()
savefig("test_multi.png")
```
![test_multi.png](test_multi.png "Plot of multidim smoothing")

## RBF Estimation

    fit_rbf(xv::Array, yv::Array, cp::Array)

fit thin-plate radial basis function according to:

    `xv` : array NxP, N number of training points, P number of input variables

    `yv` : array NxQ, N number of training points, Q number of output variables

    `cp` : array KxP, K number of control points, P number of input variables

returns a callable RBF object.

### Example

```julia
using PyPlot, KissSmoothing
t = LinRange(0,2pi,1000)
ty = sin.(t)
y = ty .+ randn(length(t)) .*0.05
fn = fit_rbf(t,y,LinRange(0,2pi,20))
scatter(t, y, color="gray",s=2,label="noisy")
plot(t, fn(t), color="red",lw=1.5,label="rbf estimate")
plot(t,ty, color="blue",lw=1.0,label="true")
xlabel("X")
ylabel("Y")
legend()
tight_layout()
savefig("rbf.png")
```
![rbf.png](rbf.png "Plot of rbf estimation")
