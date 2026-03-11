import gleam/list
import gleam/option.{type Option, None, Some}
import gleastsq/options.{
  type LeastSquareOptions, Damping, DampingDecrease, DampingIncrease, Epsilon,
  Iterations, Tolerance,
}

pub type FitParams {
  FitParams(
    iterations: Option(Int),
    epsilon: Option(Float),
    tolerance: Option(Float),
    damping: Option(Float),
    damping_increase: Option(Float),
    damping_decrease: Option(Float),
  )
}

pub fn decode_params(params: List(LeastSquareOptions)) -> FitParams {
  list.fold(
    over: params,
    from: FitParams(
      iterations: None,
      epsilon: None,
      tolerance: None,
      damping: None,
      damping_increase: None,
      damping_decrease: None,
    ),
    with: fn(acc, param) {
      case param {
        Iterations(i) -> FitParams(..acc, iterations: Some(i))
        Epsilon(e) -> FitParams(..acc, epsilon: Some(e))
        Tolerance(t) -> FitParams(..acc, tolerance: Some(t))
        Damping(d) -> FitParams(..acc, damping: Some(d))
        DampingIncrease(di) -> FitParams(..acc, damping_increase: Some(di))
        DampingDecrease(dd) -> FitParams(..acc, damping_decrease: Some(dd))
      }
    },
  )
}
