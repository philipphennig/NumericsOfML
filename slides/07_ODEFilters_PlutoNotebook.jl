### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a8fd0216-cae6-411e-ad05-aed7a13cfd03
using LinearAlgebra, Plots, ForwardDiff, PlutoUI, PlutoTeachingTools, GaussianDistributions, LaTeXStrings

# ╔═╡ a7a61543-57a9-4a4a-84f3-8f6b6d205fd0
TableOfContents()

# ╔═╡ 39d711a6-4409-49ea-968b-bb3197fee9ff
present_button()

# ╔═╡ 9d769d96-d090-4b3a-a366-8ae509cde467
md"""# Building Blocks
As in the slides."""

# ╔═╡ e55f16c7-5b3f-4e95-8dec-7b2e1618129c
function kf_predict(μ, Σ, A, Q)
    μ = A * μ
    Σ = A * Σ * A' + Q
    return μ, Σ
end

# ╔═╡ 4b3a08a8-50f9-4ad1-8e87-b7779c191c18
function kf_update(μ, Σ, H, R, y)
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ = μ + K * (y - H * μ)
    Σ = Σ - K * S * K'
    return μ, Σ
end

# ╔═╡ 88825a27-ae19-4690-afca-662352d9cbac
function ekf_update(μ, Σ, h, R, y)
    ŷ = h(μ)
    H = ForwardDiff.jacobian(h, μ)
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ = μ + K * (y - ŷ)
    Σ = Σ - K * S * K'
    return μ, Σ
end

# ╔═╡ e71bb924-e53a-432e-adae-1855579735fa
md"""
# The Extended Kalman ODE Filter
As defined on the slides
"""

# ╔═╡ 6e483977-dd56-4643-91b9-6b5ca39961d7
function extended_kalman_ode_filter(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1)
    out = []
    d = length(x₀)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    μ, Σ = kf_update(μ, Σ, E1, zeros(d, d), f(x₀, nothing))
    push!(out, (μ, Σ))

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ = ekf_update(μ_p, Σ_p, m, zeros(d, d), zeros(d))

        push!(out, (μ, Σ))
    end

    return out
end

# ╔═╡ a1129ace-c61b-4748-a6e8-634cc462c7ce
md"## Setting up the ODE Filter"

# ╔═╡ aaf4f6f4-5492-4634-96bd-d0b50443193b
md"""
### Example problem: Logistic ODE
```math
\dot{x}(t) = x(t) (1 - x(t)), \quad x(0) = 0.01,
```
Analytical solution: ``x^*(t) = 0.01 \frac{e^t}{0.01 e^t + 0.99}``.
"""

# ╔═╡ aa6628c3-761e-4ccf-b17b-0a1101168b40
f(x, t) = x .* (1 .- x)

# ╔═╡ 3ec02b0f-f593-48a9-866c-871afd803c58
x0 = [0.01]

# ╔═╡ 851edaf8-9d10-426f-8f63-799afaf4e37b
tspan = (0, 10)

# ╔═╡ 02ea96c2-245e-4d6f-9f18-1cf3dd750cc0
x_true(t) = x0[1]*exp(t) / (x0[1]*exp(t) + 1 - 0.01)

# ╔═╡ 4b406a9d-2a90-4ce5-88f9-46b740a07a76
md"The solution looks like this:"

# ╔═╡ 638afbcb-5f3b-4563-bcfe-b296d257f6ee
plot(x_true, tspan[1]:0.1:tspan[2], color=:black, linestyle=:dash, label="x_true", size=(700, 200), legend=:bottomright)

# ╔═╡ 10948ceb-d210-42f3-bc14-d9c675f3379f
md"""
### Prior
Recall:
```math
x(t+h) \mid x(t) \sim \mathcal{N} \left( A(h) x(t), Q(h) \right).
```
The prior choice (its dimensions) also defines the projection matrices ``E_0, E_1``.
"""

# ╔═╡ 9ce915f7-4c06-4792-bad9-923b58a884b1
A(h) = [
    1 h h^2/2
    0 1 h
    0 0 1
]

# ╔═╡ 83668ae1-d65c-46e9-a74e-da505105dec7
Q(h) = [
    h^5/20 h^4/8 h^3/6
    h^4/8 h^3/3 h^2/2
    h^3/6 h^2/2 h
]

# ╔═╡ ae537ddf-fe64-4c7e-9c8b-368298dbbe03
E0, E1, E2 = [1 0 0], [0 1 0], [0 0 1]

# ╔═╡ 9788e562-2764-4743-914c-8d03b32de7c5
md"### Initial Distribution"

# ╔═╡ e02134c0-4154-41e7-b1bb-289cd1beac4d
μ₀, Σ₀ = zeros(3), I(3)

# ╔═╡ 60df4149-4c1b-4bc6-bd5f-f3b15d53fae3
md"### The Discrete Time Grid"

# ╔═╡ 02019055-2b92-480f-bd71-955f4fcfc869
dt = 0.1

# ╔═╡ 1deb44ed-4247-4cec-b97e-c9726c91498e
ts = tspan[1]:dt:tspan[2]

# ╔═╡ 6c570544-8c64-4e62-9619-4331da28c348
md"## Result: A _probabilistic_ numerical ODE solution"

# ╔═╡ 7b937efd-5a38-41db-8d64-7efb80f8ed18
md"**It works!** 🥳"

# ╔═╡ 88ff7963-adae-4f31-87c3-56fc340397a6
md"## Playing around with step sizes"

# ╔═╡ 9035e571-cb1c-420c-aaa8-3f8be418e903
md"dt: $(@bind _dt Slider(10 .^ (-0.05:-0.05:-2), default=0.1, show_value=true))

smooth: $(@bind _smooth CheckBox(default=false))"

# ╔═╡ e2076b18-376b-4fc5-a8d3-add0ebc3caf2
Foldable(
    "One aspect seems not that great yet. Which one?",
    md"The uncertainties seem too large!"
)

# ╔═╡ 7294d143-ec57-48e0-8bbe-46dca05f51fd
md"""
# Uncertainty calibration
Can we improve the uncertainties from above? Yes!
By estimating a hyperparameter of the prior.
"""

# ╔═╡ 7b23bb38-8be8-4072-b50f-3ac8186ae25b
md"""
From the slides:
```math
\hat{\sigma}_N^2 = \frac{1}{Nd} \sum_{i=1}^N \left( z_i - \hat{z}_i \right)^\top S_i^{-1} \left( z_i - \hat{z}_i \right).
```
and then
```math
\Sigma_i \gets \hat{\sigma}_N^2 \cdot \Sigma_i, \quad \forall i \in \{1, \dots, N\}.
```
"""

# ╔═╡ 741ef076-7931-4edb-add5-0655c2bad97c
function ekf_update_cal(μ, Σ, h, R, y)
    ŷ = h(μ)
    H = ForwardDiff.jacobian(h, μ)
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ = μ + K * (y - ŷ)
    Σ = Σ - K * S * K'
    σ_inc = (y - ŷ)' / S * (y - ŷ) # also compute the diffusion increment
    return μ, Σ, σ_inc
end

# ╔═╡ 178945f3-719e-481a-b877-7754e50c602e
function extended_kalman_ode_filter_cal(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1)
    out = []
    d = length(x₀)
    N = length(ts)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    push!(out, (μ, Σ))
    σ̂² = 0

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ, σ_inc = ekf_update_cal(μ_p, Σ_p, m, zeros(d, d), zeros(d))

        σ̂² += σ_inc / N # Compute the diffusion estimate on-line

        push!(out, (μ, Σ))
    end
    for (_, Σ) in out
        Σ .*= σ̂² # Scale the covariances with the correct diffusion
    end

    return out
end

# ╔═╡ 18013b03-fca2-48e1-a5d8-5faa40f4284e
md"dt: $(@bind _dt_cal Slider(10 .^ (-0.05:-0.05:-2), default=0.1, show_value=true))

smooth: $(@bind _smooth_cal CheckBox(default=false))"

# ╔═╡ 17a741c3-0020-4612-b0c1-2a650af723d0
md"Looking at errors and error estimates:"

# ╔═╡ 0cd0ee64-5753-4c51-a424-c45339d6bb09
md"# Flexible Information Operators"

# ╔═╡ 1dbac200-8c18-448d-ba54-ee03efab09a4
md"""
## Solving a second-order ODE
ODE filters can also solve second-order ODEs!
We only have to adjust the measurement model, to: `m(x) = E2*x - f(E1*x, E0*x, t)`.

**Example: Harmonic oscillator**
```math
\ddot{x}(t) = - x(t), \quad x(0) = 0, \quad \dot{x}(0) = \pi/2.
```
**Analytical solution:** ``x^*(t) = \frac{\pi}{2} \sin(t)``.
"""

# ╔═╡ 7f33787e-b4ea-44a0-92d6-b790c7042245
md"**Result:**"

# ╔═╡ 6138bc25-10e2-478b-b444-5913f91d5ece
md"dt: $(@bind _dt2 Slider(10 .^ (0.3:-0.1:-2), default=0.1, show_value=true))

smooth: $(@bind _smooth2 CheckBox(default=false))"

# ╔═╡ 229c0046-16d3-4369-b75a-1112042005a6
let
    f = Foldable(
        "unfold to see the answer",
        md"""That amounts to having ``\tilde{x} := [x, \dot{x}]`` and adjusting the vector field and intial value accordingly. And indeed, mathematically the true solution to that new ODE is exactly the same as the true solution to the first-order ODE. But the probabilistic numerical solution is not identical!

        In a nutshell, ``\dot{x}`` is modeled both as ``X^{(0)}_2`` and ``X^{(1)}_1``; but in a default ODE filter, we would have two different priors for these! And we get two different posteriors, as shown here:
        """)

    md"""
   #### Why not just transform the second-order ODE to first order?
   $(f)
   """
end

# ╔═╡ 9a7d8140-994a-41fd-a2d1-dec6c6deb3ac
md"""
So the correct thing to do is really to solve the second-order ODE directly, as shown above.

That also comes with relevant speed benefits! Remember that the `EK1` scales ``O(N d^3 q^3)`` --- so transforming the ODE and thus doubling ``d`` makes things quite a bit more expensive!
"""

# ╔═╡ 3bb1a595-ba14-4e82-88de-be81a537864a
md"# Things we don't have time for today"

# ╔═╡ 20b66275-20ef-4648-930f-a57cf45ad7d6
md"""
## The `EK0` and its stability properties

"""

# ╔═╡ e141df42-b026-45e2-b4ff-ba75ab07f410
md"""### The `EK0`

The `EK0` approximates the Jacobian with the vector field as ``J_f \approx 0``.
This implies ``H := E_1``.

Pros:
- Does not need autodiff
- *Much* faster (not in the implementation in this notebook, but check out [this paper](https://proceedings.mlr.press/v162/kramer22b.html) for more)
Cons:
- Less stable (as will be shown below)
- Less expressive uncertainty estimates (always increasing)
"""

# ╔═╡ b88304f9-2ef9-457f-aa48-48090cc7f915
function ek0_update(μ, Σ, h, R, y, E1)
    ŷ = h(μ)
    H = E1
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ = μ + K * (y - ŷ)
    Σ = Σ - K * S * K'
    σ_inc = (y - ŷ)' / S * (y - ŷ)
    return μ, Σ, σ_inc
end

# ╔═╡ 41bd6af6-ce42-4627-ab47-bfb41a27d802
md"dt: $(@bind _dt_ek0 Slider(10 .^ (0.5:-0.1:-2), default=0.1, show_value=true))

smooth: $(@bind _smooth_ek0 CheckBox(default=false))"

# ╔═╡ 42a7be4f-0394-4db6-9773-cbc2fc6e3bb0
md"""### Stability
Just a quick demo of the different stability properties of the two linearization strategies.

Linear test equation
```math
\dot{x}(t) = - x(t), \quad x(0) = 1.
```
We still use a 2-times integrated Wiener process prior as above.
"""

# ╔═╡ a43a60fa-6660-42db-ac8c-d03b11441230
md"dt: $(@bind _dt_stab Slider(0.1:0.1:2, default=0.1, show_value=true))

smooth: $(@bind _smooth_stab CheckBox(default=false))"

# ╔═╡ 31fb5ea3-2f99-4b9c-9dc7-acfd04e79307
md"""
## Numerical issues and square-root filtering
When step sizes are very small the covariances can become negative-definite; which is a problem!
Fortunately, we can fix this by using the square-root filtering.
"""

# ╔═╡ c7dfc928-a998-4bc2-bb42-af33117fc492
dt_for_sqrt = 1e-10

# ╔═╡ e7f2b2a5-bb7e-4d2b-8641-fb074b1c802f
md"The current implementation (`extended_kalman_ode_smoother`) fails at those step sizes:"

# ╔═╡ c0fdf752-151d-4cae-b38f-fb1ea1a908c3
let
    ts = 0.0:dt_for_sqrt:(10*dt_for_sqrt) # just take 10 steps
    estimates = extended_kalman_ode_filter(μ₀, Σ₀, A, Q, f, x0, ts, E0, E1)
    standard_deviations = [sqrt.(diag(C)) for (m, C) in estimates]
end

# ╔═╡ 0b4aa193-7900-4bc8-8162-175256d7e356
md"**Solution: Square-root filtering**"

# ╔═╡ 07c7fb38-b7c5-4660-b6ab-559e7d639fa4
function kf_predict_sqrt(μ, ΣL, A, QL)
    μ = A * μ
    ΣL = qr([A * ΣL QL]').R'
    return μ, ΣL
end

# ╔═╡ c95b5c93-06fa-4f2b-82ec-d8a01037a35b
function kf_update_sqrt(μ, ΣL, H, RL, y)
    Σ = ΣL * ΣL'
    S = H * Σ * H' + RL * RL'
    K = Σ * H' / S
    μ = μ + K * (y - H * μ)
    ΣL = qr([(I - K * H) * ΣL K * RL]').R'
    return μ, ΣL
end

# ╔═╡ 27ce1dbe-bcc2-4e08-abef-c2ab3b57106d
md"But using square-root filters (`ek1_sqrt`) works:"

# ╔═╡ 484f7661-809c-4567-81e3-14bf64cf6ee4
md"Just to show that the implementation is still doing the right thing, here's `ek1_sqrt` in action:"

# ╔═╡ 19a23a32-1381-47a3-9348-197a83fbbd2d
md"# Final remarks

If you actullay want to solve ODEs with ODE filters, try out [ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl)! It includes calibration, smoother (i.e. higher-order) priors, uses a square-root implementation, and is generally aimed towards a very efficient implementation."

# ╔═╡ 302281a0-f349-4954-a74d-22f05d5da0bb
md"""# Utilities
In [Pluto.jl notebooks](https://github.com/fonsp/Pluto.jl), the visual order of cells is not relevant. That's very convenient, because in this way we can define utility functions, but hide it at the very end of the document!
"""

# ╔═╡ fffd13c2-f9b4-4451-8d49-3c091b14483d
md"### Logistic ODE true derivatives"

# ╔═╡ 9a49ddcf-ac79-46df-b01b-a498d6a595d8
dx_true(t) = ForwardDiff.derivative(x_true, t)

# ╔═╡ 5368de33-e3a4-4836-a908-79643dadf0ae
ddx_true(t) = ForwardDiff.derivative(dx_true, t)

# ╔═╡ 6e2df7bc-7de5-407c-8306-05fce24824d5
begin
    filter_estimates = extended_kalman_ode_filter(μ₀, Σ₀, A, Q, f, x0, ts, E0, E1)

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'

    p = plot(ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_filt",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )

    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash, ylims=(-0.1, 1.1))
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash, ylims=(-0.1, 0.35))
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash, ylims=(-0.5, 0.5))
    p
end

# ╔═╡ 87401dcb-c7c1-4a49-8ec7-45fe5e51c1ee
md"### Smoother implementation"

# ╔═╡ 014cbf78-4341-444f-8fe9-f48500c45a3d
function kf_smooth(μ, Σ, ξ, Λ, A, Q)
    μ_p = A * μ
    Σ_p = A * Σ * A' + Q
    G = Σ * A' / Σ_p
    ξ = μ + G * (ξ - μ_p)
    Λ = Σ + G * (Λ - Σ_p) * G'
    return ξ, Λ
end

# ╔═╡ e5025819-cde3-495d-9cd7-5bb0d014ac42
function extended_kalman_ode_filter_2(μ₀, Σ₀, A, Q, f, x₀, dx₀, ts, E0, E1, E2;
    smooth=true)
    out = []
    d = length(x₀)
    N = length(ts)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    # Additionally update on the first derivative:
    μ, Σ = kf_update(μ, Σ, E1, zeros(d, d), dx₀)
    push!(out, (μ, Σ))
    σ̂² = 0

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E2 * X - f(E1 * X, E0 * X, ts[i]) # Adjusted observation model
        μ, Σ, σ_inc = ekf_update_cal(μ_p, Σ_p, m, zeros(d, d), zeros(d))
        σ̂² += σ_inc / N

        push!(out, (μ, Σ))
    end
    for (_, Σ) in out
        Σ .*= σ̂²
    end
    if smooth # optional smoothing added here; can be turned on with a kwarg
        for i in length(out)-1:-1:1
            dt = ts[i+1] - ts[i]
            ξ, Λ = kf_smooth(out[i]..., out[i+1]..., A(dt), Q(dt))
            out[i][1] .= ξ
            out[i][2] .= Λ
        end
    end
    return out
end

# ╔═╡ 869f919d-8a4c-427c-b7e4-b60f82ff7fcd
let
    x₀, dx₀ = [0.0], [π / 2]
    f_harmonic_oscillator(dx, x, t) = -x

    ts = tspan[1]:_dt2:tspan[2]
    filter_estimates = extended_kalman_ode_filter_2(
        μ₀, Σ₀, A, Q, f_harmonic_oscillator, x₀, dx₀, ts, E0, E1, E2; smooth=_smooth2)

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'

    p = plot(ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_filt",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )

    x_true(t) = π / 2 * sin(t)
    dx_true(t) = ForwardDiff.derivative(x_true, t)
    ddx_true(t) = ForwardDiff.derivative(dx_true, t)
    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    p
end

# ╔═╡ 4e0c1365-f339-4444-93ee-dda851487da4
function ek0(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1; calibrate=true, smooth=true)
    out = []
    d = length(x₀)
    N = length(ts)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    push!(out, (μ, Σ))
    σ̂² = 0

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ, σ_inc = ek0_update(μ_p, Σ_p, m, zeros(d, d), zeros(d), E1)

        σ̂² += σ_inc / N

        push!(out, (μ, Σ))
    end

    if calibrate
        for (_, Σ) in out
            Σ .*= σ̂²
        end
    end

    if smooth
        for i in length(out)-1:-1:1
            dt = ts[i+1] - ts[i]
            ξ, Λ = kf_smooth(out[i]..., out[i+1]..., A(dt), Q(dt))
            out[i][1] .= ξ
            out[i][2] .= Λ
        end
    end

    return out
end

# ╔═╡ 63286e5d-00cc-4866-8fcb-3835bfd0d270
function extended_kalman_ode_smoother(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1)
    out = []
    d = length(x₀)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    push!(out, (μ, Σ))

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ = ekf_update(μ_p, Σ_p, m, zeros(d, d), zeros(d))

        push!(out, (μ, Σ))
    end

    for i in length(out)-1:-1:1
        dt = ts[i+1] - ts[i]
        ξ, Λ = kf_smooth(out[i]..., out[i+1]..., A(dt), Q(dt))
        out[i][1] .= ξ
        out[i][2] .= Λ
    end

    return out
end

# ╔═╡ c887eb98-0917-453b-bd23-42b32367b5d9
let
    _ts = tspan[1]:_dt:tspan[2]
    filter_estimates = if !_smooth
        extended_kalman_ode_filter(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    else
        extended_kalman_ode_smoother(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    end

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'
    p = plot(_ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_filt",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )
    plot!(p[1], ylims=(-0.2, 1.2))
    plot!(p[2], ylims=(-0.2, 0.7))
    plot!(p[3], ylims=(-0.3, 0.3))

    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)

    p
end

# ╔═╡ 1886feff-8038-4ead-8e14-80956ee43fc0
md"### Square-root implementation"

# ╔═╡ f7781ad8-4db3-4baa-8a84-4a6f162e22e8
function ekf_update_sqrt(μ, ΣL, h, RL, y)
    Σ = ΣL * ΣL'
    ŷ = h(μ)
    H = ForwardDiff.jacobian(h, μ)
    S = H * Σ * H' + RL * RL'
    K = Σ * H' / S
    μ = μ + K * (y - ŷ)
    ΣL = qr([(I - K * H) * ΣL K * RL]').R'
    return μ, ΣL
end

# ╔═╡ 4eb14c6c-3733-43a0-91c7-113a4b0909d7
function extended_kalman_ode_filter_sqrt(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1)
    out = []
    d = length(x₀)
    Σ₀L = cholesky(Σ₀).L

    μ, ΣL = kf_update_sqrt(μ₀, Σ₀L, E0, zeros(d, d), x₀)
    push!(out, (μ, ΣL * ΣL'))

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        QL = cholesky(Q(dt)).L
        μ_p, Σ_pL = kf_predict_sqrt(μ, ΣL, A(dt), QL)

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, ΣL = ekf_update_sqrt(μ_p, Σ_pL, m, zeros(d, d), zeros(d))

        push!(out, (μ, ΣL * ΣL'))
    end

    return out
end

# ╔═╡ 3b8c5384-9d0b-475d-badb-09c1da13374c
let
    ts = 0.0:dt_for_sqrt:(10*dt_for_sqrt) # just take 10 steps
    estimates = extended_kalman_ode_filter_sqrt(μ₀, Σ₀, A, Q, f, x0, ts, E0, E1)
    standard_deviations = [sqrt.(diag(C)) for (m, C) in estimates]
end

# ╔═╡ 587ff079-be0e-4e47-a54e-ab2d9deb490d
let
    estimates = extended_kalman_ode_filter_sqrt(μ₀, Σ₀, A, Q, f, x0, ts, E0, E1)

    means = hcat([m for (m, C) in estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'

    p = plot(layout=(3, 1))
    plot!(p, ts, means, ribbon=1.96std_devs, label="",
        fillalpha=0.2, marker=:o, markersize=2, markerstrokewidth=0.5)

    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    p
end

# ╔═╡ 07b96da7-f17a-49d2-a522-7e9fc9218773
md"### Calibration implementation"

# ╔═╡ 64db2d9f-a7fd-4055-bbd4-759e0b06af73
function kf_update_cal(μ, Σ, H, R, y)
    ŷ = H * μ
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ = μ + K * (y - ŷ)
    Σ = Σ - K * S * K'
    σ_inc = (y - ŷ)' / S * (y - ŷ)
    return μ, Σ, σ_inc
end

# ╔═╡ 921b6477-14ba-43a2-81ef-006fa29d4b3f
function extended_kalman_ode_smoother_cal(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1)
    out = []
    d = length(x₀)
    N = length(ts)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    push!(out, (μ, Σ))
    σ̂² = 0

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ, σ_inc = ekf_update_cal(μ_p, Σ_p, m, zeros(d, d), zeros(d))

        σ̂² += σ_inc / N # Compute the diffusion on-line

        push!(out, (μ, Σ))
    end

    for i in length(out)-1:-1:1
        dt = ts[i+1] - ts[i]
        ξ, Λ = kf_smooth(out[i]..., out[i+1]..., A(dt), Q(dt))
        out[i][1] .= ξ
        out[i][2] .= Λ
    end
    for (_, Σ) in out
        Σ .*= σ̂² # Scale the covariances with the correct diffusion
    end

    return out
end

# ╔═╡ 90f4351a-7396-4acb-8fef-5fddf23fc595
let
    _ts = tspan[1]:_dt_cal:tspan[2]
    filter_estimates = if !_smooth_cal
        extended_kalman_ode_filter_cal(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    else
        extended_kalman_ode_smoother_cal(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    end

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'
    p = plot(_ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_filt",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )
    plot!(p[1], ylims=(-0.2, 1.2))
    plot!(p[2], ylims=(-0.2, 0.7))
    plot!(p[3], ylims=(-0.3, 0.3))

    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    p
end

# ╔═╡ 4ea16fe9-5f85-4130-bd01-a6c97269b5b8
let
    _ts = tspan[1]:_dt_cal:tspan[2]
    filter_estimates_uncal = if !_smooth_cal
        extended_kalman_ode_filter(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    else
        extended_kalman_ode_smoother(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    end
    filter_estimates = if !_smooth_cal
        extended_kalman_ode_filter_cal(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    else
        extended_kalman_ode_smoother_cal(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1)
    end

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'
    errors = means[:, 1] .- x_true.(_ts)
    p1 = plot(_ts, zero(errors),
        ribbon=1.96std_devs[:, 1],
        marker=:o, markersize=3, markerstrokewidth=0.1,
        label="calibrated",
        ylabel="Error",
        xlabel="",
        fillalpha=0.15,
		linewidth=6,
		linealpha=0.8,
    )
	plot!(p1, _ts, 1.96std_devs[:, 1], label="", color=1, linewidth=0.5)
	plot!(p1, _ts, -1.96std_devs[:, 1], label="", color=1, linewidth=0.5)
	plot!(p1, xlims=(0,10), xticks=nothing)

    means_uncal = hcat([m for (m, C) in filter_estimates_uncal]...)'
    std_devs_uncal = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates_uncal]...)'
    errors_uncal = means_uncal[:, 1] .- x_true.(_ts)
    plot!(p1, _ts, zero(errors_uncal),
        ribbon=1.96std_devs_uncal[:, 1],
        marker=:o, markersize=2, markerstrokewidth=0.1,
        label="uncalibrated",
        #ylabel="err",
        #xlabel="",
        fillalpha=0.15,
		color=2,
		linewidth=4,
		linealpha=0.8,
    )
	plot!(p1, _ts, 1.96std_devs_uncal[:, 1], label="", color=2, linewidth=0.5)
	plot!(p1, _ts, -1.96std_devs_uncal[:, 1], label="", color=2, linewidth=0.5)
    plot!(p1, _ts, errors, color=:black, label="true error", linestyle=:dash)

    ylim = max(1.1*1.96*maximum(std_devs[:, 1]), 1.1*maximum(abs.(errors)))
    #ylim = 2maximum(abs.(errors[:,1]))
    plot!(p1, ylims=(-ylim, ylim))

	p2 = plot(_ts[2:end], abs.(errors[2:end]), color=:black, label="true error", linestyle=:dash)
	plot!(p2, _ts[2:end], std_devs[2:end, 1], label="calibrated", color=1)
	plot!(p2, _ts[2:end], std_devs_uncal[2:end, 1], label="uncalibrated", color=2)
	plot!(p2, ylabel="Error")
	plot!(p2, xlims=(0,10), xticks=[0,10])

    p = plot(p1, p2, layout=(2, 1), yscale=:log10)
end

# ╔═╡ 579ece89-06dd-427a-ab24-730044ca5b31
md"### EK1 convenience function
To have calibration and smoothing all in one function with kwargs.
"

# ╔═╡ 34f95fea-d280-4c1d-9169-8524a4904426
function ek1(μ₀, Σ₀, A, Q, f, x₀, ts, E0, E1; calibrate=true, smooth=true, initdu=false)
    out = []
    d = length(x₀)
    N = length(ts)

    μ, Σ = kf_update(μ₀, Σ₀, E0, zeros(d, d), x₀)
    if initdu
		μ, Σ = kf_update(μ, Σ, E1, zeros(d,d), f(x₀,ts[1]))
	end
    push!(out, (μ, Σ))
    σ̂² = 0

    for i in 2:length(ts)
        dt = ts[i] - ts[i-1]
        μ_p, Σ_p = kf_predict(μ, Σ, A(dt), Q(dt))

        m(X) = E1 * X - f(E0 * X, ts[i])
        μ, Σ, σ_inc = ekf_update_cal(μ_p, Σ_p, m, zeros(d, d), zeros(d))

        σ̂² += σ_inc / N

        push!(out, (μ, Σ))
    end

    if calibrate
        for (_, Σ) in out
            Σ .*= σ̂²
        end
    end

    if smooth
        for i in length(out)-1:-1:1
            dt = ts[i+1] - ts[i]
            ξ, Λ = kf_smooth(out[i]..., out[i+1]..., A(dt), Q(dt))
            out[i][1] .= ξ
            out[i][2] .= Λ
        end
    end

    return out
end

# ╔═╡ 5dd253af-fa5b-4651-aace-6211b372cc2a
let
    λ = -1
    f(x, t) = λ * x
    x0 = [1.0]
    dt = _dt_stab
    ts = tspan[1]:dt:tspan[2]
    calibrate = true
    ek0_estimates = ek0(
        μ₀, Σ₀, A, Q, f, x0, ts, E0, E1;
        calibrate=calibrate,
        smooth=_smooth_stab
    )
    ek1_estimates = ek1(
        μ₀, Σ₀, A, Q, f, x0, ts, E0, E1;
        calibrate=calibrate,
        smooth=_smooth_stab
    )

    ek0_means = hcat([m for (m, C) in ek0_estimates]...)'
    ek0_std_devs = hcat([sqrt.(diag(C)) for (m, C) in ek0_estimates]...)'
    ek1_means = hcat([m for (m, C) in ek1_estimates]...)'
    ek1_std_devs = hcat([sqrt.(diag(C)) for (m, C) in ek1_estimates]...)'

    p = plot(layout=(3, 1))
    plot!(p, ts, ek0_means, ribbon=1.96ek0_std_devs, label="EK0",
        fillalpha=0.2, marker=:o, markersize=2, markerstrokewidth=0.5)
    plot!(p, ts, ek1_means, ribbon=1.96ek1_std_devs, label="EK1",
        fillalpha=0.2, marker=:o, markersize=2, markerstrokewidth=0.5)

    plot!(p,
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        ylims=(-1.1, 1.1)
    )

    plot!(p[1], t -> exp(-t), tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], t -> -exp(-t), tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], t -> exp(-t), tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    p
end

# ╔═╡ 20fd2b48-633c-4d43-86da-98d82cdafba9
let
    x₀ = [0.0, π / 2]
    f_harmonic_oscillator(x, t) = [x[2], -x[1]]

    ts = tspan[1]:_dt2:tspan[2]
    _A(h) = kron(A(h), I(2))
    _Q(h) = kron(Q(h), I(2))
    _E0 = kron(E0, I(2))
    _E1 = kron(E1, I(2))
    μ₀, Σ₀ = zeros(2 * 3), I(2 * 3)
    filter_estimates = ek1(
        μ₀, Σ₀, _A, _Q, f_harmonic_oscillator, x₀, ts, _E0, _E1; smooth=_smooth2)

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(max.(diag(C), 0)) for (m, C) in filter_estimates]...)'

    x_0 = (means*_E0')[:, 1]
    x_0_std = (std_devs*_E0')[:, 1]
    p1 = plot(ts, x_0, ribbon=1.96x_0_std,
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label=L"X^{(0)}_1",
        ylabel=L"x(t)",
        #xlabel=L"t",
        fillalpha=0.2,
    )

    x_1_0 = (means*_E0')[:, 2]
    x_1_0_std = (std_devs*_E0')[:, 2]
    x_1_1 = (means*_E1')[:, 1]
    x_1_1_std = (std_devs*_E1')[:, 1]
    p2 = plot(ts, x_1_0, ribbon=1.96x_1_0_std,
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label=L"X^{(0)}_2",
        ylabel=L"\dot{x}(t)",
        xlabel=L"t",
        fillalpha=0.2,
    )
    plot!(ts, x_1_1, ribbon=1.96x_1_1_std,
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label=L"X^{(1)}_1",
        fillalpha=0.2,
    )
    title!(p1, "EK1 solution to the transformed ODE")

    x_true(t) = π / 2 * sin(t)
    dx_true(t) = ForwardDiff.derivative(x_true, t)
    plot!(p1, x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p2, dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)

    plot!(p1, p2, layout=(2, 1))
end

# ╔═╡ 9d6c2bc9-f816-4639-a41f-f8091cd7a089
let
    _ts = tspan[1]:_dt_ek0:tspan[2]
    filter_estimates = ek0(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1; smooth=_smooth_ek0)

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'
    p = plot(_ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_ek0",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )

	filter_estimates_ek1 = ek1(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1; smooth=_smooth_ek0)
	means = hcat([m for (m, C) in filter_estimates_ek1]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates_ek1]...)'
    plot!(p, _ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_ek1",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )

    plot!(p[1], ylims=(-0.2, 1.2))
    plot!(p[2], ylims=(-0.2, 0.7))
    plot!(p[3], ylims=(-0.3, 0.3))

    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)

    p
end

# ╔═╡ f3fa6c1b-27f1-452c-9e72-939a51b2fc1d
let
    _ts = tspan[1]:_dt_stab:tspan[2]
	f(x, t) = -x
	x0 = [1.0]
    filter_estimates = ek0(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1; smooth=_smooth_stab)

    means = hcat([m for (m, C) in filter_estimates]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates]...)'
    p = plot(_ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_ek0",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
    )

	filter_estimates_ek1 = ek1(μ₀, Σ₀, A, Q, f, x0, _ts, E0, E1; smooth=_smooth_ek0)
	means = hcat([m for (m, C) in filter_estimates_ek1]...)'
    std_devs = hcat([sqrt.(diag(C)) for (m, C) in filter_estimates_ek1]...)'
    plot!(p, _ts, means, ribbon=1.96std_devs, layout=(3, 1),
        marker=:o, markersize=2, markerstrokewidth=0.5,
        label="x_ek1",
        ylabel=[L"x(t)" L"\dot{x}(t)" L"\ddot{x}(t)"],
        xlabel=["" "" L"t"],
        fillalpha=0.2,
		size=(600, 600),
    )

    plot!(p[1], ylims=(-0.5, 1.5))
    plot!(p[2], ylims=(-1.5, 0.5))
    plot!(p[3], ylims=(-0.5, 1.5))

	x_true(t) = exp(-t)
	dx_true(t) = -exp(-t)
    ddx_true(t) = exp(-t)
    plot!(p[1], x_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[2], dx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)
    plot!(p[3], ddx_true, tspan[1]:0.01:tspan[2], label="x_true", color=:black, linestyle=:dash)

    p
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GaussianDistributions = "43dcc890-d446-5863-8d1a-14597580bb8d"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "148177ac382481e919e23fde4a1fa696f00a837f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "cc4bd91eba9cdbbb4df4746124c22c0832a460d6"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "9a0472ec2f5409db243160a8b030f94c380167a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "051072ff2accc6e0e87b708ddee39b18aa04a0bc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "501a4bf76fd679e7fcd678725d5072177392e756"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.1+0"

[[deps.GaussianDistributions]]
deps = ["Distributions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "e64d64fa3d7d69a0c7099b77d1cab82ebdbae4ac"
uuid = "43dcc890-d446-5863-8d1a-14597580bb8d"
version = "0.5.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "e1acc37ed078d99a714ed8376446f92a5535ca65"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2a64e529fd5b228224360f80f3314a633f30e0e1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.17"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "df6830e37943c7aaa10023471ca47fb3065cc3c4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5b7690dd212e026bbab1860016a6601cb077ab66"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "dadd6e31706ec493192a70a7090d369771a9a22a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.37.2"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "ea3e4ac2e49e3438815f8946fa7673b658e35bdb"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "18c35ed630d7229c5584b945641a73ca83fb5213"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.2"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "ffc098086f35909741f71ce21d03dadf0d2bfa76"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.11"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "e4bdc63f5c6d62e80eb1c0043fcc0360d5950ff7"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.10"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═a8fd0216-cae6-411e-ad05-aed7a13cfd03
# ╠═a7a61543-57a9-4a4a-84f3-8f6b6d205fd0
# ╟─39d711a6-4409-49ea-968b-bb3197fee9ff
# ╟─9d769d96-d090-4b3a-a366-8ae509cde467
# ╠═e55f16c7-5b3f-4e95-8dec-7b2e1618129c
# ╠═4b3a08a8-50f9-4ad1-8e87-b7779c191c18
# ╠═88825a27-ae19-4690-afca-662352d9cbac
# ╟─e71bb924-e53a-432e-adae-1855579735fa
# ╠═6e483977-dd56-4643-91b9-6b5ca39961d7
# ╟─a1129ace-c61b-4748-a6e8-634cc462c7ce
# ╟─aaf4f6f4-5492-4634-96bd-d0b50443193b
# ╠═aa6628c3-761e-4ccf-b17b-0a1101168b40
# ╠═3ec02b0f-f593-48a9-866c-871afd803c58
# ╠═851edaf8-9d10-426f-8f63-799afaf4e37b
# ╠═02ea96c2-245e-4d6f-9f18-1cf3dd750cc0
# ╟─4b406a9d-2a90-4ce5-88f9-46b740a07a76
# ╟─638afbcb-5f3b-4563-bcfe-b296d257f6ee
# ╟─10948ceb-d210-42f3-bc14-d9c675f3379f
# ╠═9ce915f7-4c06-4792-bad9-923b58a884b1
# ╠═83668ae1-d65c-46e9-a74e-da505105dec7
# ╠═ae537ddf-fe64-4c7e-9c8b-368298dbbe03
# ╟─9788e562-2764-4743-914c-8d03b32de7c5
# ╠═e02134c0-4154-41e7-b1bb-289cd1beac4d
# ╟─60df4149-4c1b-4bc6-bd5f-f3b15d53fae3
# ╠═02019055-2b92-480f-bd71-955f4fcfc869
# ╠═1deb44ed-4247-4cec-b97e-c9726c91498e
# ╟─6c570544-8c64-4e62-9619-4331da28c348
# ╟─6e2df7bc-7de5-407c-8306-05fce24824d5
# ╟─7b937efd-5a38-41db-8d64-7efb80f8ed18
# ╟─88ff7963-adae-4f31-87c3-56fc340397a6
# ╟─9035e571-cb1c-420c-aaa8-3f8be418e903
# ╟─c887eb98-0917-453b-bd23-42b32367b5d9
# ╟─e2076b18-376b-4fc5-a8d3-add0ebc3caf2
# ╟─7294d143-ec57-48e0-8bbe-46dca05f51fd
# ╟─7b23bb38-8be8-4072-b50f-3ac8186ae25b
# ╠═741ef076-7931-4edb-add5-0655c2bad97c
# ╠═178945f3-719e-481a-b877-7754e50c602e
# ╟─90f4351a-7396-4acb-8fef-5fddf23fc595
# ╟─18013b03-fca2-48e1-a5d8-5faa40f4284e
# ╟─17a741c3-0020-4612-b0c1-2a650af723d0
# ╟─4ea16fe9-5f85-4130-bd01-a6c97269b5b8
# ╟─5dd253af-fa5b-4651-aace-6211b372cc2a
# ╟─0cd0ee64-5753-4c51-a424-c45339d6bb09
# ╟─1dbac200-8c18-448d-ba54-ee03efab09a4
# ╟─e5025819-cde3-495d-9cd7-5bb0d014ac42
# ╟─7f33787e-b4ea-44a0-92d6-b790c7042245
# ╟─6138bc25-10e2-478b-b444-5913f91d5ece
# ╟─869f919d-8a4c-427c-b7e4-b60f82ff7fcd
# ╟─229c0046-16d3-4369-b75a-1112042005a6
# ╟─20fd2b48-633c-4d43-86da-98d82cdafba9
# ╟─9a7d8140-994a-41fd-a2d1-dec6c6deb3ac
# ╟─3bb1a595-ba14-4e82-88de-be81a537864a
# ╟─20b66275-20ef-4648-930f-a57cf45ad7d6
# ╟─e141df42-b026-45e2-b4ff-ba75ab07f410
# ╟─b88304f9-2ef9-457f-aa48-48090cc7f915
# ╟─4e0c1365-f339-4444-93ee-dda851487da4
# ╟─41bd6af6-ce42-4627-ab47-bfb41a27d802
# ╟─9d6c2bc9-f816-4639-a41f-f8091cd7a089
# ╟─42a7be4f-0394-4db6-9773-cbc2fc6e3bb0
# ╟─a43a60fa-6660-42db-ac8c-d03b11441230
# ╟─f3fa6c1b-27f1-452c-9e72-939a51b2fc1d
# ╟─31fb5ea3-2f99-4b9c-9dc7-acfd04e79307
# ╠═c7dfc928-a998-4bc2-bb42-af33117fc492
# ╟─e7f2b2a5-bb7e-4d2b-8641-fb074b1c802f
# ╠═c0fdf752-151d-4cae-b38f-fb1ea1a908c3
# ╟─0b4aa193-7900-4bc8-8162-175256d7e356
# ╟─07c7fb38-b7c5-4660-b6ab-559e7d639fa4
# ╟─c95b5c93-06fa-4f2b-82ec-d8a01037a35b
# ╟─27ce1dbe-bcc2-4e08-abef-c2ab3b57106d
# ╠═3b8c5384-9d0b-475d-badb-09c1da13374c
# ╟─484f7661-809c-4567-81e3-14bf64cf6ee4
# ╟─587ff079-be0e-4e47-a54e-ab2d9deb490d
# ╟─19a23a32-1381-47a3-9348-197a83fbbd2d
# ╟─302281a0-f349-4954-a74d-22f05d5da0bb
# ╟─fffd13c2-f9b4-4451-8d49-3c091b14483d
# ╠═9a49ddcf-ac79-46df-b01b-a498d6a595d8
# ╠═5368de33-e3a4-4836-a908-79643dadf0ae
# ╟─87401dcb-c7c1-4a49-8ec7-45fe5e51c1ee
# ╠═014cbf78-4341-444f-8fe9-f48500c45a3d
# ╠═63286e5d-00cc-4866-8fcb-3835bfd0d270
# ╟─1886feff-8038-4ead-8e14-80956ee43fc0
# ╠═f7781ad8-4db3-4baa-8a84-4a6f162e22e8
# ╠═4eb14c6c-3733-43a0-91c7-113a4b0909d7
# ╟─07b96da7-f17a-49d2-a522-7e9fc9218773
# ╠═64db2d9f-a7fd-4055-bbd4-759e0b06af73
# ╠═921b6477-14ba-43a2-81ef-006fa29d4b3f
# ╟─579ece89-06dd-427a-ab24-730044ca5b31
# ╠═34f95fea-d280-4c1d-9169-8524a4904426
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
