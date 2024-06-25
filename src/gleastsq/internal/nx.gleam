import exception
import gleam/dynamic
import gleam/result

pub type NxTensor =
  dynamic.Dynamic

type NxExceptionKeys {
  Message
}

pub type NxOpts {
  Axis(Int)
}

@external(erlang, "Elixir.Nx", "tensor")
pub fn tensor(a: List(a)) -> NxTensor

@external(erlang, "Elixir.Nx", "eye")
pub fn eye(n: Int) -> NxTensor

@external(erlang, "Elixir.Nx", "min")
pub fn min(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "max")
pub fn max(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "dot")
pub fn dot(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "multiply")
pub fn multiply(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "multiply")
pub fn multiply_mat(a: NxTensor, b: NxTensor) -> NxTensor

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

@external(erlang, "Elixir.Nx", "negate")
pub fn negate(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "shape")
pub fn shape(a: NxTensor) -> #(Int)

@external(erlang, "Elixir.Nx", "to_list")
pub fn to_list_1d(a: NxTensor) -> List(Float)

@external(erlang, "Elixir.Nx", "to_number")
pub fn to_number(a: NxTensor) -> Float

@external(erlang, "Elixir.Nx", "new_axis")
pub fn new_axis(a: NxTensor, axis: Int) -> NxTensor

@external(erlang, "Elixir.Nx", "divide")
pub fn divide(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "divide")
pub fn divide_mat(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "concatenate")
pub fn concatenate(a: List(NxTensor), opts opts: List(NxOpts)) -> NxTensor

@external(erlang, "Elixir.Nx.LinAlg", "norm")
pub fn norm(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx.LinAlg", "solve")
pub fn unsafe_solve(a: NxTensor, b: NxTensor) -> NxTensor

pub fn solve(a: NxTensor, b: NxTensor) -> Result(NxTensor, String) {
  case exception.rescue(fn() { unsafe_solve(a, b) }) {
    Ok(r) -> Ok(r)
    Error(exception.Errored(e)) -> {
      let error_msg = e |> dynamic.field(named: Message, of: dynamic.string)
      Error(result.unwrap(error_msg, "Error solving matrix"))
    }
    _ -> panic as "Unexpected error while solving matrix"
  }
}
