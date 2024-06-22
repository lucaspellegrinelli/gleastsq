import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import math_utils.{exponential, gaussian, parabola}
import sampling.{sample_around}
import test_utils.{call_leastsq, is_close}

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

pub fn noisy_gaussian_fit_test() {
  let params = [0.0, 3.0]
  let x =
    list.range(-100, 101)
    |> list.map(int.to_float)
    |> list.map(fn(x) { x /. 10.0 })
  let y = sample_around(x, gaussian, params)
  let assert Ok(result) = call_leastsq(x, y, gaussian, [1.0, 1.0])
  is_close(result, params, 0.1) |> should.be_true
}

pub fn noisy_exponential_fit_test() {
  let params = [0.1, 1.0, 0.0]
  let x =
    list.range(0, 51)
    |> list.map(int.to_float)
    |> list.map(fn(x) { x /. 10.0 })
  let y = sample_around(x, exponential, params)
  let assert Ok(result) = call_leastsq(x, y, exponential, [1.0, 1.0, 1.0])
  is_close(result, params, 0.1) |> should.be_true
}

pub fn main() {
  gleeunit.main()
}
