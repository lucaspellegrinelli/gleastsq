import gleam/float
import gleam/int
import gleam/list
import gleam_community/maths/piecewise
import gleastsq/errors.{type FitErrors}
import utils/sampling.{sample_around}

type Optimizer =
  fn(List(Float), List(Float), fn(Float, List(Float)) -> Float, List(Float)) ->
    Result(List(Float), FitErrors)

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
) -> Bool {
  let y_a = list.map(x, func(_, params_a))
  let y_b = list.map(x, func(_, params_b))

  let assert Ok(#(min_y, max_y)) = piecewise.extrema(y_a, float.compare)
  let range = max_y -. min_y

  let sum_of_squares =
    list.zip(y_a, y_b)
    |> list.fold(0.0, fn(acc, p) {
      let diff = p.0 -. p.1
      acc +. diff *. diff
    })

  let mse = sum_of_squares /. int.to_float(list.length(x))
  let assert Ok(rmse) = float.square_root(mse)
  let nrmse = rmse /. range
  nrmse <. 0.05
}
