using KissSmoothing
using Test
using Random

@testset "Goodness of Smoothing" begin
    rng = Random.MersenneTwister(1337)
    for N in [100, 200, 400, 800]
        for α in [0.01, 0.05, 0.1]
            x = LinRange(0, 1, N)
            y = identity.(x)
            n = randn(rng, length(x)) .* α
            yr = y .+ n
            ys, yn = denoise(yr)
            c1 = sum(abs2, yr .- y)
            c2 = sum(abs2, ys .- y)
            @test c2 < c1
            s1 = sum(abs2,n)
            s2 = sum(abs2,yn)
            @test abs(log2(s2/s1)) < 1
        end
    end
end

@testset "Too few points" begin
    rng = Random.MersenneTwister(1337)
    O=[1.0,2.0,10.0]
    S,N = denoise(O)
    @test all(O.==S)
    @test all(N.==0)
    @test length(S) == 3
    @test length(N) == 3
end

@testset "Infinite smoothing" begin
    O=sign.(sin.(1:1000)).+1
    S,N = denoise(O,factor=Inf)
    @test all(S .≈ 1)
    @test all(abs.(N) .≈ 1)
end
