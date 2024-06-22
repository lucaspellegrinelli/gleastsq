import gleam/option.{type Option}
import gleam/result
import gleastsq/errors.{
  type FitErrors, JacobianTaskError, NonConverged, WrongParameters,
}
import gleastsq/internal/jacobian.{jacobian}
import gleastsq/internal/nx.{type NxTensor}
import gleastsq/internal/utils.{compare_list_sizes, convert_func_params}

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
/// - `iterations` (Option(Int))
///     The maximum number of iterations to perform.
///     Default is 100.
/// - `epsilon` (Option(Float))
///     A small value to change x when calculating the derivatives for the function.
///     Default is 0.0001.
/// - `tolerance` (Option(Float))
///     The convergence tolerance.
///     Default is 0.0001.
/// - `damping` (Option(Float))
///     The initial value of the damping parameter.
///     Default is 0.0001.
/// - `damping_increase` (Option(Float))
///     The factor by which the damping parameter is increased when a step fails.
///     Default is 10.0.
/// - `damping_decrease` (Option(Float)):
///     The factor by which the damping parameter is decreased when a step succeeds.
///     Default is 0.1.
pub fn levenberg_marquardt(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  max_iterations iterations: Option(Int),
  epsilon epsilon: Option(Float),
  tolerance tolerance: Option(Float),
  damping damping: Option(Float),
  damping_increase damping_increase: Option(Float),
  damping_decrease damping_decrease: Option(Float),
) -> Result(List(Float), FitErrors) {
  use _ <- result.try(result.replace_error(
    compare_list_sizes(x, y),
    WrongParameters("x and y must have the same length"),
  ))

  let p = nx.tensor(initial_params)
  let x = nx.tensor(x)
  let y = nx.tensor(y)
  let func = convert_func_params(func)
  let iter = option.unwrap(iterations, 100)
  let eps = option.unwrap(epsilon, 0.0001)
  let tol = option.unwrap(tolerance, 0.0001)
  let reg = option.unwrap(damping, 0.0001)
  let damping_inc = option.unwrap(damping_increase, 10.0)
  let damping_dec = option.unwrap(damping_decrease, 0.1)

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
