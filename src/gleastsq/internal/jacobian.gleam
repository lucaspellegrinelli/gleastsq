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
  // Originally this was implemented by calculating a "up_f" and a "down_f" and then
  // the jacobian column was calculated as (up_f - down_f) / (2 * epsilon).
  // But since the main bottleneck of this function is calling the Gleam function in
  // every Elixir Nx's maps (because of gleastsq/internal/nx.{convert_func_params}),
  // it was decided to calculate the jacobian column as (up_f - y_fit) / epsilon where
  // "y_fit" is the result of the function with the original parameters.

  let up_params = params |> list.index_map(fn(v, idx) {
    case idx == i {
      True -> v +. epsilon
      False -> v
    }
  })

  let up_f = list.map(x, func(_, up_params)) |> nx.tensor
  nx.new_axis(nx.divide(nx.subtract(up_f, y_fit), epsilon), 1)
}
