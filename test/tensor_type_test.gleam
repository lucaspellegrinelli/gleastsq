import gleastsq/internal/nx
import gleeunit/should

pub fn get_tensor_data_test() {
  nx.tensor([1.0])
  |> nx.raw_data
  |> should.equal(Ok(<<0, 0, 128, 63>>))
}

pub fn get_tensor_shape_1_test() {
  nx.tensor([1.0, 2.0])
  |> nx.raw_shape
  |> should.equal(Ok([2]))
}

pub fn get_tensor_shape_2_test() {
  nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  |> nx.raw_shape
  |> should.equal(Ok([3, 2]))
}

pub fn get_tensor_shape_3_test() {
  nx.tensor([
    [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
    [[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]],
  ])
  |> nx.raw_shape
  |> should.equal(Ok([2, 2, 3]))
}

pub fn get_tensor_shape_4_test() {
  nx.tensor([
    [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
    [[[8.0, 9.0], [10.0, 11.0]], [[12.0, 13.0], [14.0, 15.0]]],
  ])
  |> nx.raw_shape
  |> should.equal(Ok([2, 2, 2, 2]))
}
