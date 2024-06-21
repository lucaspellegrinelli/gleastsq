import gleam/float
import gleam/list
import gleam/result
import prng/random
import prng/seed

pub fn exponential(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  let e = 2.718281828459045
  let exp = result.unwrap(float.power(e, b *. x), 0.0)
  a *. exp +. c
}

pub fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

pub fn gaussian(x: Float, params: List(Float)) -> Float {
  let assert [mu, sigma] = params
  let pi = 3.141592653589793
  let e = 2.718281828459045
  let sqrt2 = result.unwrap(float.square_root(2.0 *. pi), 0.0)
  let first = 1.0 /. { sigma *. sqrt2 }
  let second_exp =
    -0.5 *. { result.unwrap(float.power({ x -. mu } /. sigma, 2.0), 0.0) }
  let second = result.unwrap(float.power(e, second_exp), 0.0)
  first *. second
}

pub fn sample_around(
  x: List(Float),
  f: fn(Float, List(Float)) -> Float,
  params: List(Float),
) -> List(Float) {
  let seed = seed.new(42)
  let y = list.map(x, f(_, params))
  let min_y = list.fold(y, 0.0, float.min)
  let max_y = list.fold(y, 0.0, float.max)
  let noise_gen = random.float(min_y, max_y)
  y |> list.map(fn(y) { y +. random.sample(noise_gen, seed) /. 3.0 })
}
