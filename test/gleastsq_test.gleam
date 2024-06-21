import gleam/float
import gleam/function
import gleam/int
import gleam/list
import gleam/option.{None}
import gleam/result
import gleastsq
import gleeunit
import gleeunit/should

fn exponential(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  let e = 2.718281828459045
  let exp = result.unwrap(float.power(e, b *. x), 0.0)
  a *. exp +. c
}

fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

fn call_leastsq(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.least_squares(x, y, f, p, None, None, None, None)
}

fn is_close(a: List(Float), b: List(Float)) -> Bool {
  list.zip(a, b)
  |> list.map(fn(p) { float.loosely_equals(p.0, p.1, tolerating: 0.001) })
  |> list.all(function.identity)
}

pub fn perfect_power_of_2_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(2.0, x), 0.0) })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, exponential, initial)
  is_close(result, [1.0, 0.6931, 0.0]) |> should.be_true
}

pub fn perfect_power_of_3_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(3.0, x), 0.0) })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, exponential, initial)
  is_close(result, [1.0, 1.0986, 0.0]) |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 0.0, 0.0]) |> should.be_true
}

pub fn perfect_parabola_fit_with_offset_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x +. 0.1 })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 0.0, 0.1]) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x +. x })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 1.0, 0.0]) |> should.be_true
}

pub fn main() {
  gleeunit.main()
}
