import gleam/option.{None}
import gleastsq
import gleeunit/should
import utils/curves.{double_gaussian, exponential, gaussian, parabola}
import utils/helpers.{are_fits_equivalent, fit_to_curve, generate_x_axis}

pub fn lm(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.levenberg_marquardt(x, y, f, p, None, None, None, None, None, None)
}

pub fn perfect_power_of_2_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 0.693, 0.0]
  let result = fit_to_curve(x, exponential, params, lm, noisy: False)
  are_fits_equivalent(x, exponential, params, result, tol: 0.001)
  |> should.be_true
}

pub fn perfect_power_of_3_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 1.098, 0.0]
  let result = fit_to_curve(x, exponential, params, lm, noisy: False)
  are_fits_equivalent(x, exponential, params, result, tol: 0.001)
  |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 0.0, 0.0]
  let result = fit_to_curve(x, parabola, params, lm, noisy: False)
  are_fits_equivalent(x, parabola, params, result, tol: 0.001) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 1.0, 0.1]
  let result = fit_to_curve(x, parabola, params, lm, noisy: False)
  are_fits_equivalent(x, parabola, params, result, tol: 0.001) |> should.be_true
}

pub fn perfect_gaussian_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [0.1, 10.0]
  let result = fit_to_curve(x, gaussian, params, lm, noisy: True)
  are_fits_equivalent(x, gaussian, params, result, tol: 0.001) |> should.be_true
}

pub fn noisy_exponential_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [0.1, 1.0, 0.0]
  let result = fit_to_curve(x, exponential, params, lm, noisy: True)
  are_fits_equivalent(x, exponential, params, result, tol: 0.25)
  |> should.be_true
}

pub fn noisy_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [-1.4, 0.1, 0.4]
  let result = fit_to_curve(x, parabola, params, lm, noisy: True)
  are_fits_equivalent(x, parabola, params, result, tol: 0.5) |> should.be_true
}

pub fn noisy_gaussian_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [0.1, 10.0]
  let result = fit_to_curve(x, gaussian, params, lm, noisy: True)
  are_fits_equivalent(x, gaussian, params, result, tol: 0.1) |> should.be_true
}

pub fn noisy_double_gaussian_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [1.2, 0.3, 0.5, 2.5, 2.0, 1.0]
  let result = fit_to_curve(x, double_gaussian, params, lm, noisy: True)
  are_fits_equivalent(x, double_gaussian, params, result, tol: 0.1)
  |> should.be_true
}

pub fn should_error_when_x_y_different_sizes_test() {
  lm([0.0], [], parabola, []) |> should.be_error
}
