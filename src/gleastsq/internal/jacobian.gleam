import gleam/list
import gleam/otp/task
import gleam/result
import gleastsq/internal/nx.{type NxTensor, Axis}

pub opaque type JacobianError {
  JacobianTaskError
}

pub fn jacobian(
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
    Ok(jac_cols) -> Ok(nx.concatenate(jac_cols, opts: [Axis(1)]))
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
