import gleam/bool
import gleam/float
import gleam/list
import gleam/option
import gleam/result
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/params.{type FitParams}

pub fn trust_region_reflective(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: FitParams,
) -> Result(List(Float), FitErrors) {
  use <- bool.guard(
    list.length(x) != list.length(y),
    Error(WrongParameters("x and y must have the same length")),
  )

  let x = nx.tensor(x) |> nx.to_list_1d
  let y = nx.tensor(y)
  let iter = option.unwrap(opts.iterations, 100)
  let eps = option.unwrap(opts.epsilon, 0.00001)
  let tol = option.unwrap(opts.tolerance, 0.00001)
  let reg = option.unwrap(opts.damping, 0.001)
  let delta = 1.0

  use fitted <- result.try(do_trust_region_reflective(
    x,
    y,
    func,
    initial_params,
    iter,
    eps,
    tol,
    delta,
    reg,
  ))
  Ok(fitted)
}

fn ternary(cond: Bool, a: a, b: a) -> a {
  bool.guard(cond, a, fn() { b })
}

fn dogleg(j: NxTensor, g: NxTensor, b: NxTensor, delta: Float) -> NxTensor {
  let jt = nx.transpose(j)
  let jg = nx.dot(j, g)
  let g_g = nx.dot(g, g)
  let p_u_numerator = nx.negate(g_g)
  let p_u_denominator = nx.dot(g, nx.dot(jt, jg))
  let p_u = nx.multiply_mat(nx.divide_mat(p_u_numerator, p_u_denominator), g)

  let p_b = nx.negate(nx.solve(b, g))

  let p_b_norm = nx.norm(p_b) |> nx.to_number
  let p_u_norm = nx.norm(p_u) |> nx.to_number

  use <- bool.guard(p_b_norm <=. delta, p_b)
  use <- bool.guard(p_u_norm >=. delta, nx.multiply(p_u, delta /. p_u_norm))

  let p_b_u = nx.subtract(p_b, p_u)
  let assert Ok(delta_sq) = float.power(delta, 2.0)
  let assert Ok(u_norm_sq) = float.power(p_u_norm, 2.0)
  let assert Ok(d_pu_sqrt) = float.square_root(delta_sq -. u_norm_sq)
  let pb_u_norm = nx.norm(p_b_u) |> nx.to_number
  let pc_factor = d_pu_sqrt /. pb_u_norm
  let p_c = nx.add(p_u, nx.multiply(p_b_u, pc_factor))
  p_c
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

  let g_norm = nx.norm(g) |> nx.to_number
  use <- bool.guard(g_norm <. tolerance, Ok(params))

  let p = dogleg(j, g, b, delta)
  let rho = rho(x, y, func, params, p, g)

  let new_delta = case rho {
    x if x >. 0.75 -> float.max(delta, 3.0 *. nx.to_number(nx.norm(p)))
    x if x <. 0.25 -> delta *. 0.25
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
    iterations - 1,
    epsilon,
    tolerance,
    new_delta,
    lambda_reg,
  )
}
