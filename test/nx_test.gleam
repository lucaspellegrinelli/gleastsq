import gleam/float
import gleam/list
import gleastsq/internal/nx
import gleeunit/should

pub fn solve_test() {
  let a = nx.tensor([[1.0, 1.0], [2.0, 1.0]])
  let b = nx.tensor([2.0, 4.0])
  let assert Ok(result) = nx.solve(a, b)
  result
  |> nx.to_list_1d
  |> list.zip([2.0, 0.0])
  |> list.all(fn(p) { float.loosely_equals(p.0, p.1, tolerating: 0.0001) })
  |> should.equal(True)
}

pub fn solve_error_on_singular_matrix_test() {
  let a = nx.tensor([[1.0, 1.0], [0.0, 0.0]])
  let b = nx.tensor([2.0, 4.0])
  nx.solve(a, b)
  |> should.equal(Error("can't solve for singular matrix"))
}
