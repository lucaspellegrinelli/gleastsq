import gleam/bool
import gleam/float
import gleam/list
import gleam/option.{type Option}
import gleam/result
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, SolveError, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/params.{type FitParams}

const float_min = -3.4028235e38

const float_max = 3.4028235e38

/// The `trust_region_reflective` function performs the Trust Region Reflective optimization algorithm.
/// It is used to solve non-linear least squares problems. This function takes as input the data points,
/// the model function, and several optional parameters to control the optimization process.
///
/// # Parameters
/// - `x` (List(Float))
///     A list of x-values of the data points.
/// - `y` (List(Float))
///     A list of y-values of the data points.
/// - `func` (fn(Float, List(Float)) -> Float)
///     The model function that takes an x-value and a list of parameters, and returns the corresponding y-value.
/// - `initial_params` (List(Float))
///     A list of initial guesses for the parameters of the model function.
/// - `lower_bounds` (List(Float))
///     A list of lower bounds for the parameters of the model function.
/// - `upper_bounds` (List(Float))
///     A list of upper bounds for the parameters of the model function.
/// - `opts` (FitParams)
///     A record with the following fields:
///     - `iterations` (Option(Int))
///         The maximum number of iterations to perform. Default is 100.
///     - `epsilon` (Option(Float))
///         The step size used to calculate the numerical gradient. Default is 0.0001.
///     - `tolerance` (Option(Float))
///         The tolerance used to stop the optimization. Default is 0.00001.
///     - `damping` (Option(Float))
///         The damping factor used to stabilize the optimization. Default is 0.0001.
pub fn trust_region_reflective(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  lower_bounds: Option(List(Float)),
  upper_bounds: Option(List(Float)),
  opts opts: FitParams,
) -> Result(List(Float), FitErrors) {
  use <- bool.guard(
    list.length(x) != list.length(y),
    Error(WrongParameters("x and y must have the same length")),
  )

  let lower_bounds =
    option.unwrap(
      lower_bounds,
      list.repeat(float_min, list.length(initial_params)),
    )
  let upper_bounds =
    option.unwrap(
      upper_bounds,
      list.repeat(float_max, list.length(initial_params)),
    )

  use <- bool.guard(
    list.length(initial_params) != list.length(lower_bounds),
    Error(WrongParameters(
      "initial_params and lower_bounds must have the same length",
    )),
  )

  use <- bool.guard(
    list.length(initial_params) != list.length(upper_bounds),
    Error(WrongParameters(
      "initial_params and upper_bounds must have the same length",
    )),
  )

  let x = nx.tensor(x) |> nx.to_list_1d
  let y = nx.tensor(y)
  let lb = nx.tensor(lower_bounds)
  let ub = nx.tensor(upper_bounds)
  let iter = option.unwrap(opts.iterations, 100)
  let eps = option.unwrap(opts.epsilon, 0.0001)
  let tol = option.unwrap(opts.tolerance, 0.00001)
  let reg = option.unwrap(opts.damping, 0.0001)
  let delta = 1.0

  let p =
    list.zip(initial_params, lower_bounds)
    |> list.map(fn(p) { float.max(p.0, p.1) })
    |> list.zip(upper_bounds)
    |> list.map(fn(p) { float.min(p.0, p.1) })

  do_trust_region_reflective(x, y, func, p, lb, ub, iter, eps, tol, delta, reg)
}

fn ternary(cond: Bool, a: a, b: a) -> a {
  bool.guard(cond, a, fn() { b })
}

fn dogleg(
  j: NxTensor,
  g: NxTensor,
  b: NxTensor,
  delta: Float,
) -> Result(NxTensor, FitErrors) {
  let jt = nx.transpose(j)
  let jg = nx.dot(j, g)
  let p_u_numerator = nx.negate(nx.dot(g, g))
  let p_u_denominator = nx.dot(g, nx.dot(jt, jg))
  let p_u = nx.multiply_mat(nx.divide_mat(p_u_numerator, p_u_denominator), g)

  use bg_solve <- result.try(result.map_error(nx.solve(b, g), SolveError))
  let p_b = nx.negate(bg_solve)

  let p_b_norm = nx.norm(p_b) |> nx.to_number
  let p_u_norm = nx.norm(p_u) |> nx.to_number

  use <- bool.guard(p_b_norm <=. delta, Ok(p_b))
  use <- bool.guard(p_u_norm >=. delta, Ok(nx.multiply(p_u, delta /. p_u_norm)))

  let p_b_u = nx.subtract(p_b, p_u)
  let assert Ok(delta_sq) = float.power(delta, 2.0)
  let assert Ok(u_norm_sq) = float.power(p_u_norm, 2.0)
  let assert Ok(d_pu_sqrt) = float.square_root(delta_sq -. u_norm_sq)
  let pb_u_norm = nx.norm(p_b_u) |> nx.to_number
  let pc_factor = d_pu_sqrt /. pb_u_norm
  let p_c = nx.add(p_u, nx.multiply(p_b_u, pc_factor))
  Ok(p_c)
}

fn dogleg_impose_bounds(
  p: NxTensor,
  params: NxTensor,
  lb: NxTensor,
  ub: NxTensor,
) -> NxTensor {
  nx.add(params, p) |> nx.min(ub) |> nx.max(lb) |> nx.subtract(params)
}

pub fn rho(
  x: List(Float),
  y: NxTensor,
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  p: NxTensor,
  g: NxTensor,
) -> Float {
  let fx = list.map(x, func(_, params)) |> nx.tensor

  let offset_params =
    list.zip(params, nx.to_list_1d(p))
    |> list.map(fn(p) { p.0 +. p.1 })

  let fx_p = list.map(x, func(_, offset_params)) |> nx.tensor

  let fx_diff = nx.pow(nx.subtract(fx, y), 2.0)
  let fxp_diff = nx.pow(nx.subtract(fx_p, y), 2.0)

  let actual_reduction_sum =
    nx.sum(nx.subtract(fx_diff, fxp_diff)) |> nx.to_number
  let actual_reduction = 0.5 *. actual_reduction_sum

  let g_dot_p = nx.dot(g, p) |> nx.to_number
  let predicted_reduction = -0.5 *. g_dot_p

  actual_reduction /. predicted_reduction
}

fn do_trust_region_reflective(
  x: List(Float),
  y: NxTensor,
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  lower_bounds: NxTensor,
  upper_bounds: NxTensor,
  iterations: Int,
  epsilon: Float,
  tolerance: Float,
  delta: Float,
  lambda_reg: Float,
) {
  use <- bool.guard(iterations == 0, Error(NonConverged))
  let m = list.length(params)

  let f = list.map(x, func(_, params)) |> nx.tensor
  let r = nx.subtract(f, y)
  use j <- result.try(result.replace_error(
    jacobian(x, f, func, params, epsilon),
    JacobianTaskError,
  ))

  let lambda_eye = nx.eye(m) |> nx.multiply(lambda_reg)
  let jt = nx.transpose(j)
  let b = nx.add(nx.dot(jt, j), lambda_eye)
  let g = nx.dot(jt, r)

  let g_norm = nx.reduce_max(nx.abs(g)) |> nx.to_number
  use <- bool.guard(g_norm <. tolerance, Ok(params))

  use non_bounded_p <- result.try(dogleg(j, g, b, delta))
  let p =
    dogleg_impose_bounds(
      non_bounded_p,
      nx.tensor(params),
      lower_bounds,
      upper_bounds,
    )
  let rho = rho(x, y, func, params, p, g)

  let p_norm = nx.norm(p) |> nx.to_number
  use <- bool.guard(p_norm <. tolerance, Ok(params))

  let new_delta = case rho {
    x if x >. 0.75 -> float.max(delta, 2.0 *. nx.to_number(nx.norm(p)))
    x if x <. 0.25 -> delta *. 0.5
    _ -> delta
  }

  let new_params =
    list.zip(params, nx.to_list_1d(p))
    |> list.map(fn(p) { p.0 +. p.1 })

  do_trust_region_reflective(
    x,
    y,
    func,
    ternary(rho >. 0.0, new_params, params),
    lower_bounds,
    upper_bounds,
    iterations - 1,
    epsilon,
    tolerance,
    new_delta,
    lambda_reg,
  )
}
