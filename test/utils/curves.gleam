import gleam/float
import gleam_community/maths/elementary

const e = 2.718281828459045

const pi = 3.141592653589793

pub fn exponential(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. elementary.exponential(b *. x) +. c
}

pub fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

pub fn gaussian(x: Float, params: List(Float)) -> Float {
  let assert [mu, sigma] = params
  let assert Ok(sqrt2) = float.square_root(2.0 *. pi)
  let first = 1.0 /. { sigma *. sqrt2 }
  let assert Ok(second_exp) = float.power({ x -. mu } /. sigma, 2.0)
  let assert Ok(second) = float.power(e, -0.5 *. second_exp)
  first *. second
}

pub fn double_gaussian(x: Float, params: List(Float)) -> Float {
  let assert [mul1, mu1, sigma1, mul2, mu2, sigma2] = params
  mul1 *. gaussian(x, [mu1, sigma1]) +. mul2 *. gaussian(x, [mu2, sigma2])
}

pub fn triple_gaussian(x: Float, params: List(Float)) -> Float {
  let assert [mul1, mu1, sigma1, mul2, mu2, sigma2, mul3, mu3, sigma3] = params
  mul1
  *. gaussian(x, [mu1, sigma1])
  +. mul2
  *. gaussian(x, [mu2, sigma2])
  +. mul3
  *. gaussian(x, [mu3, sigma3])
}
