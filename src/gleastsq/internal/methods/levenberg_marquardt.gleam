import gleam/bool
import gleam/list
import gleam/option
import gleam/result
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/params.{type FitParams}

/// The `levenberg_marquardt` function performs the Levenberg-Marquardt optimization algorithm.
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
/// - `opts` (FitParams)
///     A record with the following fields:
///     - `iterations` (Option(Int))
///         The maximum number of iterations to perform. Default is 100.
///     - `epsilon` (Option(Float))
///         The step size used to calculate the numerical gradient. Default is 0.0001.
///     - `tolerance` (Option(Float))
///         The tolerance used to stop the optimization. Default is 0.0001.
///     - `damping` (Option(Float))
///         The damping factor used to stabilize the optimization. Default is 0.0001.
///     - `damping_increase` (Option(Float))
///         The factor used to increase the damping factor when the optimization is not improving. Default is 10.0.
///     - `damping_decrease` (Option(Float))
///         The factor used to decrease the damping factor when the optimization is improving. Default is 0.1.
pub fn levenberg_marquardt(
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

  let p = nx.tensor(initial_params)
  let x = nx.tensor(x)
  let y = nx.tensor(y)
  let func = nx.convert_func_params(func)
  let iter = option.unwrap(opts.iterations, 100)
  let eps = option.unwrap(opts.epsilon, 0.0001)
  let tol = option.unwrap(opts.tolerance, 0.0001)
  let reg = option.unwrap(opts.damping, 0.0001)
  let damping_inc = option.unwrap(opts.damping_increase, 10.0)
  let damping_dec = option.unwrap(opts.damping_decrease, 0.1)

  use fitted <- result.try(do_levenberg_marquardt(
    x,
    y,
    func,
    p,
    iter,
    eps,
    tol,
    reg,
    damping_inc,
    damping_dec,
  ))
  Ok(fitted |> nx.to_list_1d)
}

fn ternary(cond: Bool, a: a, b: a) -> a {
  case cond {
    True -> a
    False -> b
  }
}

fn do_levenberg_marquardt(
  x: NxTensor,
  y: NxTensor,
  func: fn(NxTensor, NxTensor) -> Float,
  params: NxTensor,
  max_iterations: Int,
  epsilon: Float,
  tolerance: Float,
  damping: Float,
  damping_increase: Float,
  damping_decrease: Float,
) -> Result(NxTensor, FitErrors) {
  let m = nx.shape(params).0
  case max_iterations {
    0 -> Error(NonConverged)
    iterations -> {
      let r = x |> nx.map(func(_, params)) |> nx.subtract(y, _)
      use j <- result.try(result.replace_error(
        jacobian(x, func, params, epsilon),
        JacobianTaskError,
      ))

      let jt = nx.transpose(j)
      let lambda_eye = nx.eye(m) |> nx.multiply(damping)
      let h_damped = nx.add(nx.dot(jt, j), lambda_eye)
      let g = nx.dot(jt, r)
      let delta = nx.solve(h_damped, g)

      let new_params = nx.add(params, delta)
      case nx.to_number(nx.norm(delta)) {
        x if x <. tolerance -> Ok(new_params)
        _ -> {
          let new_r = x |> nx.map(func(_, new_params)) |> nx.subtract(y, _)
          let prev_error = nx.sum(nx.pow(r, 2.0)) |> nx.to_number
          let new_error = nx.sum(nx.pow(new_r, 2.0)) |> nx.to_number
          let impr = new_error <. prev_error
          do_levenberg_marquardt(
            x,
            y,
            func,
            ternary(impr, new_params, params),
            iterations - 1,
            epsilon,
            tolerance,
            damping *. ternary(impr, damping_decrease, damping_increase),
            damping_increase,
            damping_decrease,
          )
        }
      }
    }
  }
}
