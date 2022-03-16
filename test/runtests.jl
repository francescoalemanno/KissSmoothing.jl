using KissSmoothing
using Test
using Random
@testset "Optimizers" begin
    rng = Random.MersenneTwister(1337)
    for N in [100, 200, 400, 800]
        for α in [0.01, 0.05, 0.1]
            x = LinRange(0, 1, N)
            y = x
            yr = y .+ randn(rng, length(x)) .* α
            ys, yn = denoise(yr)
            c1 = sum(abs2, yr .- y)
            c2 = sum(abs2, ys .- y)
            @test c2 < c1
        end
    end
end
