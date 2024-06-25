import exception
import gleam/dynamic
import gleam/list
import gleam/result

pub type NxTensor =
  dynamic.Dynamic

type NxTensorKeys {
  Data
  State
  Shape
}

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

@external(erlang, "Elixir.Nx.LinAlg", "solve")
pub fn unsafe_solve(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx.LinAlg", "norm")
pub fn norm(a: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "new_axis")
pub fn new_axis(a: NxTensor, axis: Int) -> NxTensor

@external(erlang, "Elixir.Nx", "divide")
pub fn divide(a: NxTensor, b: Float) -> NxTensor

@external(erlang, "Elixir.Nx", "divide")
pub fn divide_mat(a: NxTensor, b: NxTensor) -> NxTensor

@external(erlang, "Elixir.Nx", "concatenate")
pub fn concatenate(a: List(NxTensor), opts opts: List(NxOpts)) -> NxTensor

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

pub fn raw_data(a: NxTensor) -> Result(BitArray, Nil) {
  use data <- result.try(result.replace_error(
    a |> dynamic.field(named: Data, of: dynamic.dynamic),
    Nil,
  ))

  result.replace_error(
    data |> dynamic.field(named: State, of: dynamic.bit_array),
    Nil,
  )
}

pub fn raw_shape(a: NxTensor) -> Result(List(Int), Nil) {
  use shape <- result.try(result.replace_error(
    a |> dynamic.field(named: Shape, of: dynamic.dynamic),
    Nil,
  ))

  Ok(do_raw_shape(shape, 0, []))
}

fn do_raw_shape(shape: dynamic.Dynamic, index: Int, acc: List(Int)) -> List(Int) {
  let element = shape |> dynamic.element(index, dynamic.int)
  case element {
    Ok(e) -> do_raw_shape(shape, index + 1, list.append(acc, [e]))
    Error(_) -> acc
  }
}
