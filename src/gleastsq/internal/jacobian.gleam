import gleam/list
import gleam/otp/task
import gleam/result
import gleastsq/internal/nx.{type NxTensor, Axis}

pub opaque type JacobianError {
  JacobianTaskError
}

pub fn jacobian(
  x: List(Float),
  y_fit: NxTensor,
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  epsilon: Float,
) {
  let n = list.length(params)
  let jac_result =
    list.range(0, n - 1)
    |> list.map(fn(i) {
      task.async(fn() {
        compute_jacobian_col(x, y_fit, func, params, epsilon, i)
      })
    })
    |> list.map(task.try_await_forever(_))
    |> result.all

  case jac_result {
    Ok(jac_cols) -> Ok(nx.concatenate(jac_cols, opts: [Axis(1)]))
    Error(_) -> Error(JacobianTaskError)
  }
}

fn compute_jacobian_col(
  x: List(Float),
  y_fit: NxTensor,
  func: fn(Float, List(Float)) -> Float,
  params: List(Float),
  epsilon: Float,
  i: Int,
) -> NxTensor {
  let up_params =
    list.index_map(params, fn(v, idx) {
      case idx == i {
        True -> v +. epsilon
        False -> v
      }
    })

  let up_f = list.map(x, func(_, up_params)) |> nx.tensor
  nx.new_axis(nx.divide(nx.subtract(up_f, y_fit), epsilon), 1)
}
