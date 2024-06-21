# gleastsq

[![Package Version](https://img.shields.io/hexpm/v/gleastsq)](https://hex.pm/packages/gleastsq)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/gleastsq/)

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

## Development

```sh
gleam test  # Run the tests
gleam shell # Run an Erlang shell
```
