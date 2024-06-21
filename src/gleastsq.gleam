import gleam/option.{type Option, None, Some}
import gleam/result
import nx.{type NxTensor}

pub opaque type FitErrors {
  NonConverged
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
  let #(m) = nx.shape(x)
  let jac = nx.broadcast(0.0, #(m, n))
  compute_jacobian(x, func, params, jac, epsilon, n, 0)
}

fn compute_jacobian(
  x: NxTensor,
  func: fn(NxTensor, NxTensor) -> Float,
  params: NxTensor,
  jac: NxTensor,
  epsilon: Float,
  n: Int,
  i: Int,
) {
  case i {
    i if i >= n -> jac
    _ -> {
      let mask =
        nx.indexed_put(nx.broadcast(0.0, #(n)), nx.tensor([i]), epsilon)
      let up_params = nx.add(params, mask)
      let down_params = nx.subtract(params, mask)

      let up_f = nx.map(x, func(_, up_params))
      let down_f = nx.map(x, func(_, down_params))
      let deriv =
        nx.new_axis(nx.divide(nx.subtract(up_f, down_f), 2.0 *. epsilon), 1)

      let updated_jac = nx.put_slice(jac, [0, i], deriv)

      compute_jacobian(x, func, params, updated_jac, epsilon, n, i + 1)
    }
  }
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
      let j = jacobian(x, func, params, epsilon)
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

  let reg = case lambda_reg {
    Some(x) -> x
    None -> 0.0001
  }

  let tol = case tolerance {
    Some(x) -> x
    None -> 0.0001
  }

  use fitted <- result.try(do_least_squares(x, y, func, p, iter, eps, tol, reg))
  Ok(fitted |> nx.to_list_1d)
}
