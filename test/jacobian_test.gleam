import gleam/float
import gleam/int
import gleam/list
import gleastsq/internal/jacobian
import gleastsq/internal/nx
import gleeunit/should

fn affine(x: Float, params: List(Float)) -> Float {
  let assert [a, b] = params
  a *. x +. b
}

fn unit_vector(length: Int, index: Int) -> List(Float) {
  int.range(from: 0, to: length, with: [], run: fn(acc, i) {
    [
      case i == index {
        True -> 1.0
        False -> 0.0
      },
      ..acc
    ]
  })
  |> list.reverse
}

fn approx_equal_list(left: List(Float), right: List(Float)) -> Bool {
  list.zip(left, right)
  |> list.all(fn(pair) {
    float.loosely_equals(pair.0, pair.1, tolerating: 0.001)
  })
}

fn jacobian_column(j: nx.NxTensor, width: Int, index: Int) -> List(Float) {
  nx.dot(j, nx.tensor(unit_vector(width, index))) |> nx.to_list_1d
}

pub fn jacobian_matches_affine_model_derivatives_test() {
  let x = [-1.0, 0.0, 2.0]
  let params = [3.0, 4.0]
  let y_fit = list.map(x, affine(_, params)) |> nx.tensor
  let assert Ok(j) = jacobian.jacobian(x, y_fit, affine, params, 0.01)

  jacobian_column(j, 2, 0)
  |> approx_equal_list(x)
  |> should.be_true

  jacobian_column(j, 2, 1)
  |> approx_equal_list([1.0, 1.0, 1.0])
  |> should.be_true
}

pub fn jacobian_preserves_parameter_column_order_test() {
  let x = [2.0, 4.0]
  let params = [1.5, -3.0]
  let y_fit = list.map(x, affine(_, params)) |> nx.tensor
  let assert Ok(j) = jacobian.jacobian(x, y_fit, affine, params, 0.01)

  jacobian_column(j, 2, 0)
  |> approx_equal_list([2.0, 4.0])
  |> should.be_true

  jacobian_column(j, 2, 1)
  |> approx_equal_list([1.0, 1.0])
  |> should.be_true
}
