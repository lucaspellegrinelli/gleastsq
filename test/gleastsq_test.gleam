import gleam/float
import gleam/int
import gleam/list
import gleam/option.{None}
import gleam/result
import gleeunit
import gleeunit/should
import gleastsq

const e = 2.718281828459045

fn exponential(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  let exp = result.unwrap(float.power(e, b *. x), 0.0)
  a *. exp +. c
}

fn parabola(x: Float, params: List(Float)) -> Float {
  let assert [a, b, c] = params
  a *. x *. x +. b *. x +. c
}

fn is_close(a: List(Float), b: List(Float)) -> Bool {
  let residual =
    list.zip(a, b)
    |> list.fold(0.0, fn(acc, p) { acc +. float.absolute_value(p.0 -. p.1) })

  residual <. 0.001
}

pub fn main() {
  gleeunit.main()
}

pub fn perfect_power_of_2_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(2.0, x), 0.0) })

  let initial_guess = [1.0, 1.0, 1.0]
  let result =
    gleastsq.least_squares(
      x,
      y,
      exponential,
      initial_guess,
      None,
      None,
      None,
      None,
    )

  is_close(result, [1.0, 0.6931, 0.0]) |> should.be_true
}

pub fn perfect_power_of_3_fit_test() {
  let x = list.range(0, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { result.unwrap(float.power(3.0, x), 0.0) })

  let initial_guess = [1.0, 1.0, 1.0]
  let result =
    gleastsq.least_squares(
      x,
      y,
      exponential,
      initial_guess,
      None,
      None,
      None,
      None,
    )

  is_close(result, [1.0, 1.0986, 0.0]) |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = list.range(-5, 6) |> list.map(int.to_float)
  let y = list.map(x, fn(x) { x *. x })

  let initial_guess = [1.0, 1.0, 1.0]
  let result =
    gleastsq.least_squares(x, y, parabola, initial_guess, None, None, None, None)

  is_close(result, [1.0, 0.0, 0.0]) |> should.be_true
}
