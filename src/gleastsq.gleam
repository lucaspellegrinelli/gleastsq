import gleam/option.{type Option}
import gleastsq/least_squares as lsqr
import gleastsq/levemberg_marquardt as lm

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
) {
  lm.levenberg_marquardt(
    x,
    y,
    func,
    initial_params,
    iterations,
    epsilon,
    tolerance,
    damping,
    damping_increase,
    damping_decrease,
  )
}

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
/// - `iterations` (Option(Int))
///     The maximum number of iterations to perform.
///     Default is 100.
/// - `epsilon` (Option(Float))
///     A small value to change x when calculating the derivatives for the function.
///     Default is 0.0001.
/// - `tolerance` (Option(Float))
///     The convergence tolerance.
///     Default is 0.0001.
/// - `lambda_reg` (Option(Float))
///     The regularization parameter.
///     Default is 0.0001.
pub fn least_squares(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  max_iterations iterations: Option(Int),
  epsilon epsilon: Option(Float),
  tolerance tolerance: Option(Float),
  lambda_reg lambda_reg: Option(Float),
) {
  lsqr.least_squares(
    x,
    y,
    func,
    initial_params,
    iterations,
    epsilon,
    tolerance,
    lambda_reg,
  )
}
