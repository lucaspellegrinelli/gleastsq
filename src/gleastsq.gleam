import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/otp/task
import gleam/result
import gleastsq/internal/nx.{type NxTensor}

pub opaque type FitErrors {
  NonConverged
  WrongParameters(String)
  JacobianTaskError
}

fn convert_func_params(
  func: fn(Float, List(Float)) -> Float,
) -> fn(NxTensor, NxTensor) -> Float {
  fn(x: NxTensor, params: NxTensor) -> Float {
    func(nx.to_number(x), nx.to_list_1d(params))
  }
}

fn jacobian(
  x: NxTensor,
  func: fn(NxTensor, NxTensor) -> Float,
  params: NxTensor,
  epsilon: Float,
) {
  let #(n) = nx.shape(params)
  let jac_result =
    list.range(0, n - 1)
    |> list.map(fn(i) {
      task.async(fn() { compute_jacobian_col(x, func, params, epsilon, n, i) })
    })
    |> list.map(task.try_await_forever(_))
    |> result.all

  case jac_result {
    Ok(jac_cols) -> Ok(nx.concatenate(jac_cols, 1))
    Error(_) -> Error(JacobianTaskError)
  }
}

fn compute_jacobian_col(
  x: NxTensor,
  func: fn(NxTensor, NxTensor) -> Float,
  params: NxTensor,
  epsilon: Float,
  n: Int,
  i: Int,
) -> NxTensor {
  let mask = nx.indexed_put(nx.broadcast(0.0, #(n)), nx.tensor([i]), epsilon)
  let up_params = nx.add(params, mask)
  let down_params = nx.subtract(params, mask)
  let up_f = nx.map(x, func(_, up_params))
  let down_f = nx.map(x, func(_, down_params))
  nx.new_axis(nx.divide(nx.subtract(up_f, down_f), 2.0 *. epsilon), 1)
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
      use j <- result.try(jacobian(x, func, params, epsilon))
      let jt = nx.transpose(j)
      let lambda_eye = nx.eye(m) |> nx.multiply(lambda_reg)
      let h = nx.add(nx.dot(jt, j), lambda_eye)
      let g = nx.dot(jt, r)
      let delta = nx.solve(h, g)

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

fn compare_list_sizes(
  a: List(a),
  b: List(a),
  msg: String,
  callback: fn() -> Result(b, FitErrors),
) {
  case list.length(a) == list.length(b) {
    True -> callback()
    False -> Error(WrongParameters(msg))
  }
}

/// Compute the least squares fit of a function to a set of data points.
///
/// ## Parameters:
/// - `x`: The X axis of the data points as a list of floats.
/// - `y`: The Y axis of the data points as a list of floats.
/// - `func`: The function to fit to the data points. The function should take a float and a list of floats (the function coefficients) as arguments and return a float.
/// - `initial_params`: The initial guess for the function coefficients.
/// - `max_iterations`: The maximum number of iterations to perform. Default is 100.
/// - `epsilon`: The epsilon value for the numerical derivative. Default is 0.0001.
/// - `tolerance`: The tolerance for the convergence criterion. Default is 0.0001.
/// - `lambda_reg`: The regularization parameter. Default is 0.0001.
///
/// ## Examples
///
/// ```gleam
/// fn parabola(x: Float, params: List(Float)) -> Float {
///   let assert [a, b, c] = params
///   a *. x *. x +. b *. x +. c
/// }
/// 
/// pub fn main() {
///   let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
///   let y = list.map(x, fn(x) { x *. x })
///   let initial_guess = [1.0, 1.0, 1.0]
/// 
///   let assert Ok(result) =
///     gleastsq.least_squares(
///       x,
///       y,
///       parabola,
///       initial_guess,
///       max_iterations: None,
///       epsilon: None,
///       tolerance: None,
///       lambda_reg: None,
///     )
/// 
///   io.debug(result) // [1.0, 0.0, 0.0] (within numerical error)
/// }
/// ```
pub fn least_squares(
  x: List(Float),
  y: List(Float),
  func: fn(Float, List(Float)) -> Float,
  initial_params: List(Float),
  max_iterations iterations: Option(Int),
  epsilon epsilon: Option(Float),
  tolerance tolerance: Option(Float),
  lambda_reg lambda_reg: Option(Float),
) -> Result(List(Float), FitErrors) {
  use <- compare_list_sizes(x, y, "x and y must have the same length")

  let p = nx.tensor(initial_params)
  let x = nx.tensor(x)
  let y = nx.tensor(y)
  let func = convert_func_params(func)

  let iter = case iterations {
    Some(x) -> x
    None -> 100
  }

  let eps = case epsilon {
    Some(x) -> x
    None -> 0.0001
  }

  let tol = case tolerance {
    Some(x) -> x
    None -> 0.0001
  }

  let reg = case lambda_reg {
    Some(x) -> x
    None -> 0.0001
  }

  use fitted <- result.try(do_least_squares(x, y, func, p, iter, eps, tol, reg))
  Ok(fitted |> nx.to_list_1d)
}
