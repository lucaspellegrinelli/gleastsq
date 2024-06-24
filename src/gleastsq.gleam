import gleastsq/internal/methods/gauss_newton as gn
import gleastsq/internal/methods/levenberg_marquardt as lm
import gleastsq/internal/methods/trust_region_reflective as trr
import gleastsq/internal/params.{decode_params}
import gleastsq/options.{type LeastSquareOptions}

/// The `least_squares` function is an alias for the `levenberg_marquardt` function.
/// Check the documentation of the `levenberg_marquardt` function for more information.
pub fn least_squares(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: List(LeastSquareOptions),
) {
  lm.levenberg_marquardt(x, y, func, initial_params, decode_params(opts))
}

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
/// - `opts` (List(LeastSquareOptions))
///     A list of optional parameters to control the optimization process.
///     The available options are:
///     - `Iterations(Int)`: The maximum number of iterations to perform. Default is 100.
///     - `Epsilon(Float)`: A small value to change x when calculating the derivatives for the function. Default is 0.0001.
///     - `Tolerance(Float)`: The convergence tolerance. Default is 0.0001.
///     - `Damping(Float)`: The initial value of the damping parameter. Default is 0.0001.
///     - `DampingIncrease(Float)`: The factor by which the damping parameter is increased when a step fails. Default is 10.0.
///     - `DampingDecrease(Float)`: The factor by which the damping parameter is decreased when a step succeeds. Default is 0.1.
///
/// # Example
/// ```gleam
/// import gleam/io
/// import gleastsq
/// import gleastsq/options.{Iterations, Tolerance}
///
/// fn parabola(x: Float, params: List(Float)) -> Float {
///   let assert [a, b, c] = params
///   a *. x *. x +. b *. x +. c
/// }
///
/// pub fn main() {
///   let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
///   let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
///   let initial_guess = [1.0, 1.0, 1.0]
///
///   let assert Ok(result) =
///     gleastsq.levenberg_marquardt(
///       x,
///       y,
///       parabola,
///       initial_guess,
///       opts: [Iterations(1000), Tolerance(0.001)]
///     )
///
///   io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
/// }
/// ```
pub fn levenberg_marquardt(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: List(LeastSquareOptions),
) {
  lm.levenberg_marquardt(x, y, func, initial_params, decode_params(opts))
}

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
/// - `opts` (List(LeastSquareOptions))
///     A list of optional parameters to control the optimization process.
///     The available options are:
///     - `Iterations(Int)`: The maximum number of iterations to perform. Default is 100.
///     - `Epsilon(Float)`: A small value to change x when calculating the derivatives for the function. Default is 0.0001.
///     - `Tolerance(Float)`: The convergence tolerance. Default is 0.0001.
///     - `Damping(Float)`: The value of the damping parameter. Default is 0.001.
///
/// # Example
/// ```gleam
/// import gleam/io
/// import gleastsq
/// import gleastsq/options.{Iterations, Tolerance}
///
/// fn parabola(x: Float, params: List(Float)) -> Float {
///   let assert [a, b, c] = params
///   a *. x *. x +. b *. x +. c
/// }
///
/// pub fn main() {
///   let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
///   let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
///   let initial_guess = [1.0, 1.0, 1.0]
///
///   let assert Ok(result) =
///     gleastsq.gauss_newton(
///       x,
///       y,
///       parabola,
///       initial_guess,
///       opts: [Iterations(1000), Tolerance(0.001)]
///     )
///
///   io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
/// }
/// ```
pub fn gauss_newton(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: List(LeastSquareOptions),
) {
  gn.gauss_newton(x, y, func, initial_params, decode_params(opts))
}

/// The `trust_region_reflective` function performs a least squares optimization using the Trust Region Reflective algorithm.
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
/// - `opts` (List(LeastSquareOptions))
///     A list of optional parameters to control the optimization process.
///     The available options are:
///     - `Iterations(Int)`: The maximum number of iterations to perform. Default is 100.
///     - `Epsilon(Float)`: A small value to change x when calculating the derivatives for the function. Default is 0.0001.
///     - `Tolerance(Float)`: The convergence tolerance. Default is 0.0001.
///     - `Damping(Float)`: The value of the damping parameter. Default is 0.001.
///
/// # Example
/// ```gleam
/// import gleam/io
/// import gleastsq
/// import gleastsq/options.{Iterations, Tolerance}
///
/// fn parabola(x: Float, params: List(Float)) -> Float {
///   let assert [a, b, c] = params
///   a *. x *. x +. b *. x +. c
/// }
///
/// pub fn main() {
///   let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
///   let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
///   let initial_guess = [1.0, 1.0, 1.0]
///
///   let assert Ok(result) =
///     gleastsq.trust_region_reflective(
///       x,
///       y,
///       parabola,
///       initial_guess,
///       opts: [Iterations(1000), Tolerance(0.001)]
///     )
///
///   io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
/// }
/// ```
pub fn trust_region_reflective(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  opts opts: List(LeastSquareOptions),
) {
  trr.trust_region_reflective(x, y, func, initial_params, decode_params(opts))
}
