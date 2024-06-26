import gleam/bool
import gleam/list
import gleam/option
import gleam/result
import gleam_community/maths/metrics.{norm}
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, SolveError, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/params.{type FitParams}

/// The `gauss_newton` function performs a basic least squares optimization algorithm.
/// It is used to find the best-fit parameters for a given model function to a set of data points.
/// This function takes as input the data points, the model function, and several optional parameters to control the optimization process.
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
/// - `opts` (FitParams)
///     A record with the following fields:
///     - `iterations` (Option(Int))
///         The maximum number of iterations to perform. Default is 100.
///     - `epsilon` (Option(Float))
///         The step size used to calculate the numerical gradient. Default is 0.0001.
///     - `tolerance` (Option(Float))
///         The tolerance used to stop the optimization. Default is 0.0001.
///     - `damping` (Option(Float))
///         The damping factor used to stabilize the optimization. Default is 0.001.
pub fn gauss_newton(
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
  let reg = option.unwrap(opts.damping, 0.001)

  do_gauss_newton(x, y, func, initial_params, iter, eps, tol, reg)
}

fn do_gauss_newton(
  x: List(Float),
  y: NxTensor,
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  iterations: Int,
  epsilon: Float,
  tolerance: Float,
  lambda_reg: Float,
) {
  use <- bool.guard(iterations == 0, Error(NonConverged))

  let m = list.length(params)
  let y_fit = list.map(x, func(_, params)) |> nx.tensor
  let r = nx.subtract(y, y_fit)
  use j <- result.try(result.replace_error(
    jacobian(x, y_fit, func, params, epsilon),
    JacobianTaskError,
  ))

  let jt = nx.transpose(j)
  let eye = nx.eye(m) |> nx.multiply(lambda_reg)
  let jtj = nx.add(nx.dot(jt, j), eye)
  let jt_r = nx.dot(jt, r)

  use delta_solve <- result.try(result.map_error(
    nx.solve(jtj, jt_r),
    SolveError,
  ))
  let delta = delta_solve |> nx.to_list_1d
  let delta_norm = norm(delta, 2.0)

  let new_params =
    list.zip(params, delta)
    |> list.map(fn(p) { p.0 +. p.1 })

  use <- bool.guard(delta_norm <. tolerance, Ok(new_params))

  do_gauss_newton(
    x,
    y,
    func,
    new_params,
    iterations - 1,
    epsilon,
    tolerance,
    lambda_reg,
  )
}
