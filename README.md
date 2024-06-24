# gleastsq

[![Package Version](https://img.shields.io/hexpm/v/gleastsq)](https://hex.pm/packages/gleastsq)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/gleastsq/)

A least squares curve fitting library for Gleam. This library uses the [Nx](https://hexdocs.pm/nx/Nx.html)
library from Elixir under the hood to perform matrix operations.

## Which method should I use?

The library provides four functions for curve fitting: `least_squares`, `gauss_newton`, `levenberg_marquardt` and `trust_region_reflective`.

### Least Squares

The `least_squares` function is just an alias for the `levenberg_marquardt` function.

## Levenberg-Marquardt

Ideal for non-linear least squares problems, particularly when the initial guess is far from the solution. It combines the benefits of the Gauss-Newton method and gradient descent, making it robust and efficient for various scenarios. However, it requires careful tuning of the damping parameter to balance convergence speed and stability.

## Trust-Region Reflective

Best suited for large-scale problems or those with constraints, this method ensures that each iteration stays within a predefined "trust region," preventing large, unstable steps. It is reliable and effective for challenging optimization problems but can be computationally intensive.

## Gauss-Newton

Efficient for problems where residuals are small and the initial guess is close to the true solution. It approximates the Hessian matrix, leading to faster convergence for well-behaved problems. However, it may struggle with highly non-linear problems or poor initial guesses, as it lacks the robustness of the Levenberg-Marquardt and trust-region reflective methods.

## Installation

```sh
gleam add gleastsq
```

```gleam
import gleam/io
import gleastsq

fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

pub fn main() {
  let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
  let initial_guess = [1.0, 1.0, 1.0]

  let assert Ok(result) =
    gleastsq.least_squares(x, y, parabola, initial_guess, opts: [])

  io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
}
```

Further documentation can be found at <https://hexdocs.pm/gleastsq>.
