import gleam/option
import gleam/result
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/params.{type FitParams}
import gleastsq/internal/utils.{compare_list_sizes, convert_func_params}

/// The `least_squares` function performs a basic least squares optimization algorithm.
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
///         The damping factor used to stabilize the optimization. Default is 0.0001.
pub fn least_squares(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: FitParams,
) -> Result(List(Float), FitErrors) {
  use _ <- result.try(result.replace_error(
    compare_list_sizes(x, y),
    WrongParameters("x and y must have the same length"),
  ))

  let p = nx.tensor(initial_params)
  let x = nx.tensor(x)
  let y = nx.tensor(y)
  let func = convert_func_params(func)
  let iter = option.unwrap(opts.iterations, 100)
  let eps = option.unwrap(opts.epsilon, 0.0001)
  let tol = option.unwrap(opts.tolerance, 0.0001)
  let reg = option.unwrap(opts.damping, 0.0001)

  use fitted <- result.try(do_least_squares(x, y, func, p, iter, eps, tol, reg))
  Ok(fitted |> nx.to_list_1d)
}

fn do_least_squares(
  x: NxTensor,
  y: NxTensor,
  func: fn(NxTensor, NxTensor) -> Float,
  params: NxTensor,
  max_iterations: Int,
  epsilon: Float,
  tolerance: Float,
  lambda_reg: Float,
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
      let eye = nx.eye(m) |> nx.multiply(lambda_reg)
      let jtj = nx.add(nx.dot(jt, j), eye)
      let jt_r = nx.dot(jt, r)
      let delta = nx.solve(jtj, jt_r)

      case nx.to_number(nx.norm(delta)) {
        x if x <. tolerance -> Ok(params)
        _ ->
          do_least_squares(
            x,
            y,
            func,
            nx.add(params, delta),
            iterations - 1,
            epsilon,
            tolerance,
            lambda_reg,
          )
      }
    }
  }
}
