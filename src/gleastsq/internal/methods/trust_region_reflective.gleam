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
  let pu1 = nx.negate(nx.dot(g, g))
  let pu2 = nx.dot(g, nx.dot(jt, nx.dot(j, g)))
  let pu = nx.multiply_mat(nx.divide_mat(pu1, pu2), g)
  let pu_norm = nx.norm(pu) |> nx.to_number

  let pb = nx.negate(nx.solve(b, g))
  let pb_norm = nx.norm(pb) |> nx.to_number

  use <- bool.guard(pb_norm <=. delta, pb)
  use <- bool.guard(pu_norm >=. delta, nx.multiply(pu, delta /. pu_norm))

  let pbu = nx.subtract(pb, pu)

  let assert Ok(delta_sq) = float.power(delta, 2.0)
  let assert Ok(pu_norm_sq) = float.power(pu_norm, 2.0)
  let assert Ok(d_pu_sqrt) = float.square_root(delta_sq -. pu_norm_sq)
  let pc_factor = d_pu_sqrt /. nx.to_number(nx.norm(pbu))

  nx.add(pu, nx.multiply(pbu, pc_factor))
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
  let xp = nx.add(nx.tensor(params), p) |> nx.to_list_1d
  let fxp = list.map(x, func(_, xp)) |> nx.tensor

  let fx_r = nx.pow(nx.subtract(fx, y), 2.0)
  let fxp_r = nx.pow(nx.subtract(fxp, y), 2.0)
  let mse = nx.sum(nx.subtract(fx_r, fxp_r)) |> nx.to_number
  let actual_reduction = 0.5 *. mse

  let gp = nx.dot(g, p) |> nx.to_number
  let predicted_reduction = -0.5 *. gp
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

  let jt = nx.transpose(j)
  let lambda_eye = nx.eye(m) |> nx.multiply(lambda_reg)
  let b = nx.add(nx.dot(jt, j), lambda_eye)
  let g = nx.dot(jt, r)

  let g_norm = nx.norm(g) |> nx.to_number
  use <- bool.guard(g_norm <=. tolerance, Ok(params))

  let p = dogleg(j, g, b, delta)
  let rho = rho(x, y, func, params, p, g)

  let p_norm = nx.norm(p) |> nx.to_number
  let new_delta = case rho {
    x if x >. 0.75 -> float.max(delta, 3.0 *. p_norm)
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
