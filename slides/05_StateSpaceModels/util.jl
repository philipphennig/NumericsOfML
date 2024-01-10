function simulate_linear(
    Φ,
    Q,
    u,
    H,
    R,
    v,
    μ₀,
    Σ₀,
    N::Int;
    rng = Random.GLOBAL_RNG,
)
    x = rand(rng, MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(rng, MvNormal(Φ * states[end] + u, Q)))
        push!(observations, rand(rng, MvNormal(H * states[end] + v, R)))
    end
    return states, observations
end


function simulate_nonlinear(
    f::Function,
    Q,
    h::Function,
    R,
    μ₀,
    Σ₀,
    N::Int;
    rng = Random.GLOBAL_RNG,
)
    x = rand(rng, MvNormal(μ₀, Σ₀))
    states = [x]
    observations = []

    for i in 1:N
        push!(states, rand(rng, MvNormal(f(states[end]), Q)))
        push!(observations, rand(rng, Normal(h(states[end]), R)))
    end
    return states, observations
end


stack(x) = copy(reduce(hcat, x)')


nothing