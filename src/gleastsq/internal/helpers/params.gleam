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
  do_decode_lm_params(
    params,
    FitParams(
      iterations: None,
      epsilon: None,
      tolerance: None,
      damping: None,
      damping_increase: None,
      damping_decrease: None,
    ),
  )
}

fn do_decode_lm_params(
  params: List(LeastSquareOptions),
  acc: FitParams,
) -> FitParams {
  case params {
    [] -> acc
    [Iterations(i), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, iterations: Some(i)))
    [Epsilon(e), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, epsilon: Some(e)))
    [Tolerance(t), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, tolerance: Some(t)))
    [Damping(d), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, damping: Some(d)))
    [DampingIncrease(di), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, damping_increase: Some(di)))
    [DampingDecrease(dd), ..rest] ->
      do_decode_lm_params(rest, FitParams(..acc, damping_decrease: Some(dd)))
  }
}
