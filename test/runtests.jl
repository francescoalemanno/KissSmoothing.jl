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
            @test s1*0.5 < s2 < s1*2
        end
    end
end
