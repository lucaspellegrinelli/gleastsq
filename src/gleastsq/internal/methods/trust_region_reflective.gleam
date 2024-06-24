import gleam/bool
import gleam/float
import gleam/list
import gleam/option
import gleam/result
import gleam_community/maths/metrics.{norm}
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
  let eps = option.unwrap(opts.epsilon, 0.0001)
  let tol = option.unwrap(opts.tolerance, 0.0001)
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
  ))
  Ok(fitted)
}

fn ternary(cond: Bool, a: a, b: a) -> a {
  bool.guard(cond, a, fn() { b })
}

fn dogleg(b: NxTensor, g: NxTensor, delta: Float) -> NxTensor {
  let gt = nx.transpose(g)
  let pu_1 = nx.dot(gt, g) |> nx.negate
  let pu_2 = nx.dot(nx.dot(gt, b), g)
  let pu = nx.divide_mat(pu_1, pu_2) |> nx.multiply_mat(g)
  let pu_norm = nx.norm(pu) |> nx.to_number
  use <- bool.guard(pu_norm >=. delta, nx.multiply(pu, delta /. pu_norm))

  let pb = nx.solve(b, g) |> nx.negate
  let pb_norm = nx.norm(pb) |> nx.to_number
  use <- bool.guard(pb_norm <=. delta, pb)

  let pb_pu = nx.subtract(pb, pu)
  let pb_pu_norm = nx.norm(pb_pu) |> nx.to_number
  let pu_pb = nx.multiply(pb_pu, { delta -. pu_norm } /. pb_pu_norm)
  nx.add(pu, pu_pb)
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
) {
  use <- bool.guard(iterations == 0, Error(NonConverged))
  let tp = nx.tensor(params)

  let list_f = list.map(x, func(_, params))
  let f_norm = norm(list_f, 2.0)
  let f = list_f |> nx.tensor

  use j <- result.try(result.replace_error(
    jacobian(x, f, func, params, epsilon),
    JacobianTaskError,
  ))

  let jt = nx.transpose(j)
  let g = nx.dot(jt, f)
  let b = nx.dot(jt, j)
  let ngt = nx.negate(nx.transpose(g))

  let pb = dogleg(b, g, delta)
  let pbt = nx.transpose(pb)

  let p = nx.add(tp, pb) |> nx.to_list_1d
  let fp_norm = list.map(x, func(_, p)) |> norm(2.0)

  let rho1 = nx.dot(ngt, pb) |> nx.to_number
  let rho2 = nx.dot(nx.dot(pbt, b), pb) |> nx.to_number
  let rho = { f_norm -. fp_norm } /. { rho1 -. 0.5 *. rho2 }

  use <- bool.guard(
    rho >. 0.75,
    do_trust_region_reflective(
      x,
      y,
      func,
      p,
      iterations - 1,
      epsilon,
      tolerance,
      float.min(2.0 *. delta, 1.0),
    ),
  )

  use <- bool.guard(
    rho <. 0.25,
    do_trust_region_reflective(
      x,
      y,
      func,
      p,
      iterations - 1,
      epsilon,
      tolerance,
      delta /. 2.0,
    ),
  )

  let stop_crit = tolerance *. { norm(p, 2.0) +. tolerance }
  let pb_norm = norm(nx.to_list_1d(pb), 2.0)
  use <- bool.guard(pb_norm <. stop_crit, Ok(p))

  do_trust_region_reflective(
    x,
    y,
    func,
    p,
    iterations - 1,
    epsilon,
    tolerance,
    delta,
  )
}
