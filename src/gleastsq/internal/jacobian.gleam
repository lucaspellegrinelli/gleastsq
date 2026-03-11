import exception
import gleam/erlang/process
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
    n -> {
      let jac_result =
        int.range(from: 0, to: n, with: [], run: fn(acc, i) { [i, ..acc] })
        |> list.reverse
        |> parallel_map(compute_jacobian_col(x, y_fit, func, params, epsilon, _))

      case jac_result {
        Ok(jac_cols) -> Ok(nx.concatenate(jac_cols, opts: [Axis(1)]))
        Error(_) -> Error(JacobianTaskError)
      }
    }
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

fn parallel_map(
  items: List(a),
  using mapper: fn(a) -> b,
) -> Result(List(b), Nil) {
  let results = process.new_subject()

  items
  |> list.index_map(fn(item, index) {
    process.spawn_unlinked(fn() {
      let result = case exception.rescue(fn() { mapper(item) }) {
        Ok(value) -> Ok(value)
        Error(_) -> Error(Nil)
      }

      process.send(results, #(index, result))
    })
  })

  gather_results(results, list.length(items), [])
}

fn gather_results(
  results: process.Subject(#(Int, Result(a, Nil))),
  remaining: Int,
  acc: List(#(Int, a)),
) -> Result(List(a), Nil) {
  case remaining {
    0 ->
      acc
      |> list.sort(fn(left, right) { int.compare(left.0, right.0) })
      |> list.map(fn(result) { result.1 })
      |> Ok

    _ ->
      case process.receive_forever(results) {
        #(index, Ok(value)) ->
          gather_results(results, remaining - 1, [#(index, value), ..acc])

        #(_, Error(Nil)) -> Error(Nil)
      }
  }
}
