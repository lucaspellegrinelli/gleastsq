import gleam/float
import gleam/function
import gleam/int
import gleam/list
import gleam/option.{None}
import gleam_community/maths/metrics.{mean}
import gleastsq
import gleastsq/errors.{type FitErrors}
import utils/sampling.{sample_around}

pub type Optimizer =
  fn(List(Float), List(Float), fn(Float, List(Float)) -> Float, List(Float)) ->
    Result(List(Float), FitErrors)

pub fn leastsq(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.least_squares(x, y, f, p, None, None, None, None)
}

pub fn lm(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.levenberg_marquardt(x, y, f, p, None, None, None, None, None, None)
}

pub fn generate_x_axis(from: Int, to: Int, n: Int) -> List(Float) {
  let n = int.to_float(n - 1)
  let from = int.to_float(from)
  let to = int.to_float(to)
  let range = to -. from
  let factor = n /. range
  list.range(0, float.truncate(n))
  |> list.map(int.to_float)
  |> list.map(fn(x) { x /. factor +. from })
}

pub fn fit_to_curve(
  x: List(Float),
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  optimizer: Optimizer,
  noisy noisy: Bool,
) {
  let y = case noisy {
    True -> sample_around(x, func, params)
    False -> list.map(x, func(_, params))
  }
  let initial = list.repeat(1.0, list.length(params))
  let assert Ok(result) = optimizer(x, y, func, initial)
  result
}

pub fn are_fits_equivalent(
  x: List(Float),
  func: fn(Float, List(Float)) -> Float,
  params_a: List(Float),
  params_b: List(Float),
  tol tol: Float,
) -> Bool {
  let y_a = list.map(x, func(_, params_a))
  let y_b = list.map(x, func(_, params_b))
  let assert Ok(diff) =
    list.zip(y_a, y_b)
    |> list.map(fn(p) { float.absolute_value(p.0 -. p.1) })
    |> mean

  diff <. tol
}

pub fn is_close(a: List(Float), b: List(Float), t: Float) -> Bool {
  list.zip(a, b)
  |> list.map(fn(p) { float.loosely_equals(p.0, p.1, tolerating: t) })
  |> list.all(function.identity)
}
