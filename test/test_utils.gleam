import gleam/float
import gleam/function
import gleam/list
import gleam/option.{None}
import gleastsq

pub fn call_leastsq(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.least_squares(x, y, f, p, None, None, None, None)
}

pub fn is_close(a: List(Float), b: List(Float), t: Float) -> Bool {
  list.zip(a, b)
  |> list.map(fn(p) { float.loosely_equals(p.0, p.1, tolerating: t) })
  |> list.all(function.identity)
}
