pub type NxTensor =
  Nil

pub type NxOpts {
  Axis(Int)
}

@external(erlang, "Elixir.Nx", "tensor")
pub fn tensor(a: List(a)) -> NxTensor

@external(erlang, "Elixir.Nx", "eye")
pub fn eye(n: Int) -> NxTensor

@external(erlang, "Elixir.Nx", "dot")
pub fn dot(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "map")
pub fn map(a: NxTensor, f: fn(NxTensor) -> Float) -> NxTensor

@external(erlang, "Elixir.Nx", "multiply")
pub fn multiply(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "add")
pub fn add(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "sum")
pub fn sum(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "pow")
pub fn pow(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "subtract")
pub fn subtract(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "transpose")
pub fn transpose(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "shape")
pub fn shape(a: NxTensor) -> #(Int)

@external(erlang, "Elixir.Nx", "to_list")
pub fn to_list_1d(a: NxTensor) -> List(Float)

@external(erlang, "Elixir.Nx", "to_number")
pub fn to_number(a: NxTensor) -> Float

@external(erlang, "Elixir.Nx.LinAlg", "norm")
pub fn norm(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx.LinAlg", "solve")
pub fn solve(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "broadcast")
pub fn broadcast(a: Float, shape: a) -> NxTensor

@external(erlang, "Elixir.Nx", "indexed_put")
pub fn indexed_put(a: NxTensor, indices: NxTensor, value: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "new_axis")
pub fn new_axis(a: NxTensor, axis: Int) -> NxTensor

@external(erlang, "Elixir.Nx", "divide")
pub fn divide(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "put_slice")
pub fn put_slice(a: NxTensor, indices: List(Int), value: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "concatenate")
pub fn concatenate(a: List(NxTensor), opts opts: List(NxOpts)) -> NxTensor

pub fn convert_func_params(
  func: fn(Float, List(Float)) -> Float,
) -> fn(NxTensor, NxTensor) -> Float {
  fn(x: NxTensor, params: NxTensor) -> Float {
    func(to_number(x), to_list_1d(params))
  }
}
