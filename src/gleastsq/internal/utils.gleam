import gleam/list
import gleastsq/internal/nx.{type NxTensor}

pub fn compare_list_sizes(a: List(a), b: List(a)) {
  case list.length(a) == list.length(b) {
    True -> Ok(Nil)
    False -> Error(Nil)
  }
}

pub fn convert_func_params(
  func: fn(Float, List(Float)) -> Float,
) -> fn(NxTensor, NxTensor) -> Float {
  fn(x: NxTensor, params: NxTensor) -> Float {
    func(nx.to_number(x), nx.to_list_1d(params))
  }
}
