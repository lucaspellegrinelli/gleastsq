# gleastsq

[![Package Version](https://img.shields.io/hexpm/v/gleastsq)](https://hex.pm/packages/gleastsq)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/gleastsq/)

A least squares curve fitting library for Gleam. This library uses the [Nx](https://hexdocs.pm/nx/Nx.html)
library from Elixir under the hood to perform matrix operations.

## Levenberg-Marquardt vs Leasts Squares for curve fitting

The library provides three functions for curve fitting: `least_squares`, `gauss_newton` and `levenberg_marquardt`.

### Least Squares

The `least_squares` function is just an alias for the `levenberg_marquardt` function.

### Gauss-Newton

The `gauss_newton` function is best for least squares problems with good initial guesses and small residuals.
It is less computationally intensive and thus can be faster than the Levenberg-Marquardt method but can be unstable
with poor initial guesses or large residuals.

### Levenberg-Marquardt

The `levenberg_marquardt` function is robust for nonlinear least squares problems, handling large residuals and poor
initial guesses effectively. It is more computationally intensive but provides reliable convergence for a wider range
of problems, especially in challenging or ill-conditioned cases.

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
