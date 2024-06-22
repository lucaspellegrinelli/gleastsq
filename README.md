# gleastsq

[![Package Version](https://img.shields.io/hexpm/v/gleastsq)](https://hex.pm/packages/gleastsq)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/gleastsq/)

A curve fitting library for Gleam. This library uses the [Nx](https://hexdocs.pm/nx/Nx.html) library from Elixir to perform matrix operations.

## Levemberg-Marquardt vs Leasts Squares for curve fitting

The library provides two functions for curve fitting: `least_squares` and `levemberg_marquardt`.

### Least Squares

The 'least_squares' function is generally simpler and faster but may not converge for some functions, specially for non-linear functions.
It is generally recommended for simpler models where the relationship between the parameters and the function is linear.

### Levenberg-Marquardt

The 'levemberg_marquardt' function is more robust but may be slower due to the extra calculations.
It is generally recommended for non-linear functions where the relationship between the parameters and the function is non-linear.

```sh
gleam add gleastsq
```

```gleam
import gleam/option.{None}
import gleam/list
import gleam/io
import gleastsq

fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

pub fn main() {
  let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  let y = list.map(x, fn(x) { x *. x })
  let initial_guess = [1.0, 1.0, 1.0]

  let assert Ok(result) =
    gleastsq.least_squares(
      x,
      y,
      parabola,
      initial_guess,
      max_iterations: None,
      epsilon: None,
      tolerance: None,
      lambda_reg: None,
    )

  io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
}
```

Further documentation can be found at <https://hexdocs.pm/gleastsq>.
