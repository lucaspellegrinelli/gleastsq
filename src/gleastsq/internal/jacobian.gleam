import gleam/int
import gleam/list
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
  case list.length(params) {
    0 -> Error(JacobianTaskError)
    n ->
      int.range(from: 0, to: n, with: [], run: fn(acc, i) {
        [compute_jacobian_col(x, y_fit, func, params, epsilon, i), ..acc]
      })
      |> list.reverse
      |> nx.concatenate(opts: [Axis(1)])
      |> Ok
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
