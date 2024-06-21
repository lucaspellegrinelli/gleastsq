import gleam/float
import gleam/function
import gleam/int
import gleam/list
import gleam/option.{None}
import gleam/result
import gleastsq
import gleeunit
import gleeunit/should
import math_utils.{exponential, gaussian, parabola, sample_around}

fn call_leastsq(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.least_squares(x, y, f, p, None, None, None, None)
}

fn is_close(a: List(Float), b: List(Float), t: Float) -> Bool {
  list.zip(a, b)
  |> list.map(fn(p) { float.loosely_equals(p.0, p.1, tolerating: t) })
  |> list.all(function.identity)
}

pub fn perfect_power_of_2_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(2.0, x), 0.0) })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, exponential, initial)
  is_close(result, [1.0, 0.6931, 0.0], 0.001) |> should.be_true
}

pub fn perfect_power_of_3_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(3.0, x), 0.0) })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, exponential, initial)
  is_close(result, [1.0, 1.0986, 0.0], 0.001) |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 0.0, 0.0], 0.001) |> should.be_true
}

pub fn perfect_parabola_fit_with_offset_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x +. 0.1 })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 0.0, 0.1], 0.001) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x +. x })
  let initial = [1.0, 1.0, 0.0]
  let assert Ok(result) = call_leastsq(x, y, parabola, initial)
  is_close(result, [1.0, 1.0, 0.0], 0.001) |> should.be_true
}

pub fn perfect_gaussian_fit_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, gaussian(_, [0.1, 10.0]))
  let initial = [1.0, 1.0]
  let assert Ok(result) = call_leastsq(x, y, gaussian, initial)
  is_close(result, [0.1, 10.0], 0.001) |> should.be_true
}

pub fn imperfect_gaussian_fit_test() {
  let x = list.range(-50, 51) |> list.map(int.to_float)
  let y = sample_around(x, gaussian, [0.1, 10.0])
  let initial = [1.0, 1.0]
  let assert Ok(result) = call_leastsq(x, y, gaussian, initial)
  is_close(result, [0.1, 10.0], 0.1) |> should.be_true
}

pub fn main() {
  gleeunit.main()
}
