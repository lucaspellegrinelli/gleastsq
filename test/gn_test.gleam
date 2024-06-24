import gleastsq
import gleeunit/should
import utils/curves.{
  double_gaussian, exponential, gaussian, parabola, triple_gaussian,
}
import utils/helpers.{are_fits_equivalent, fit_to_curve, generate_x_axis}

pub fn gn(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.gauss_newton(x, y, f, p, [])
}

pub fn perfect_power_of_2_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 0.693, 0.0]
  let assert Ok(result) = fit_to_curve(x, exponential, params, gn, noisy: False)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn perfect_power_of_3_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 1.098, 0.0]
  let assert Ok(result) = fit_to_curve(x, exponential, params, gn, noisy: False)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 0.0, 0.0]
  let assert Ok(result) = fit_to_curve(x, parabola, params, gn, noisy: False)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 1.0, 0.1]
  let assert Ok(result) = fit_to_curve(x, parabola, params, gn, noisy: False)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn perfect_gaussian_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [0.1, 10.0]
  let assert Ok(result) = fit_to_curve(x, gaussian, params, gn, noisy: False)
  are_fits_equivalent(x, gaussian, params, result) |> should.be_true
}

pub fn noisy_exponential_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [0.1, 1.0, 0.0]
  let assert Ok(result) = fit_to_curve(x, exponential, params, gn, noisy: True)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn noisy_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [-1.4, 0.1, 0.4]
  let assert Ok(result) = fit_to_curve(x, parabola, params, gn, noisy: True)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn noisy_gaussian_fit_test() {
  let x = generate_x_axis(-50, 50, 100)
  let params = [0.1, 10.0]
  let assert Ok(result) = fit_to_curve(x, gaussian, params, gn, noisy: True)
  are_fits_equivalent(x, gaussian, params, result) |> should.be_true
}

pub fn noisy_double_gaussian_fit_test() {
  // Gauss-Newton will not generally converge on this function
  let x = generate_x_axis(-3, 7, 100)
  let params = [1.2, 0.3, 0.5, 2.5, 2.0, 1.0]
  let result = fit_to_curve(x, double_gaussian, params, gn, noisy: True)
  case result {
    Ok(result) -> {
      // If it converges, it should be a bad fit
      are_fits_equivalent(x, double_gaussian, params, result)
      |> should.be_false
    }
    Error(_) -> {
      // We expect it to not converge
      should.be_true(True)
    }
  }
}

pub fn noisy_triple_gaussian_fit_test() {
  // Gauss-Newton will not generally converge on this function
  let x = generate_x_axis(-3, 7, 100)
  let params = [1.2, 0.3, 0.5, 2.5, 2.0, 1.0, 1.0, -2.0, 0.1]
  let result = fit_to_curve(x, triple_gaussian, params, gn, noisy: True)
  case result {
    Ok(result) -> {
      // If it converges, it should be a bad fit
      are_fits_equivalent(x, triple_gaussian, params, result)
      |> should.be_false
    }
    Error(_) -> {
      // We expect it to not converge
      should.be_true(True)
    }
  }
}

pub fn should_error_when_x_y_different_sizes_test() {
  gn([0.0], [], parabola, []) |> should.be_error
}
