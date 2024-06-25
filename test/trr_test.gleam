import gleam/list
import gleam/option.{None, Some}
import gleastsq
import gleeunit/should
import utils/curves.{
  double_gaussian, exponential, gaussian, parabola, triple_gaussian,
}
import utils/helpers.{are_fits_equivalent, fit_to_curve, generate_x_axis}

pub fn trr(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.trust_region_reflective(x, y, f, p, None, None, [])
}

pub fn perfect_power_of_2_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 0.693, 0.0]
  let assert Ok(result) =
    fit_to_curve(x, exponential, params, trr, noisy: False)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn perfect_power_of_2_fit_empty_bounds_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 0.693, 0.0]
  let y = list.map(x, exponential(_, params))
  let assert Ok(result) =
    gleastsq.trust_region_reflective(
      x,
      y,
      exponential,
      [1.0, 1.0, 1.0],
      lower_bounds: None,
      upper_bounds: None,
      opts: [],
    )
  are_fits_equivalent(x, exponential, params, result) |> should.be_true
}

pub fn perfect_power_of_2_fit_fail_if_impossible_bounds_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 0.693, 0.0]
  let y = list.map(x, exponential(_, params))
  let result =
    gleastsq.trust_region_reflective(
      x,
      y,
      exponential,
      [1.0, 1.0, 1.0],
      lower_bounds: Some([-1.0, -1.0, -1.0]),
      upper_bounds: Some([0.0, 0.0, 0.0]),
      opts: [],
    )

  case result {
    Ok(r) -> are_fits_equivalent(x, exponential, params, r) |> should.be_false
    _ -> should.be_true(True)
  }
}

pub fn perfect_power_of_3_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [1.0, 1.098, 0.0]
  let assert Ok(result) =
    fit_to_curve(x, exponential, params, trr, noisy: False)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn perfect_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 0.0, 0.0]
  let assert Ok(result) = fit_to_curve(x, parabola, params, trr, noisy: False)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 1.0, 0.1]
  let assert Ok(result) = fit_to_curve(x, parabola, params, trr, noisy: False)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn perfect_parabola_fit_with_slope_fail_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 1.0, 0.1]
  let y = list.map(x, parabola(_, params))
  let result =
    gleastsq.trust_region_reflective(
      x,
      y,
      parabola,
      [1.0, 1.0, 1.0],
      lower_bounds: Some([0.0, 0.0, 0.5]),
      upper_bounds: Some([9.0, 9.0, 9.0]),
      opts: [],
    )

  case result {
    Ok(r) -> are_fits_equivalent(x, parabola, params, r) |> should.be_false
    _ -> should.be_true(True)
  }
}

pub fn perfect_gaussian_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [0.1, 10.0]
  let assert Ok(result) = fit_to_curve(x, gaussian, params, trr, noisy: True)
  are_fits_equivalent(x, gaussian, params, result) |> should.be_true
}

pub fn perfect_gaussian_fit_fail_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [0.1, 10.0]
  let y = list.map(x, gaussian(_, params))
  let result =
    gleastsq.trust_region_reflective(
      x,
      y,
      gaussian,
      [0.1, 10.0],
      lower_bounds: Some([0.0, 0.0]),
      upper_bounds: Some([0.0, 0.0]),
      opts: [],
    )

  case result {
    Ok(r) -> are_fits_equivalent(x, gaussian, params, r) |> should.be_false
    _ -> should.be_true(True)
  }
}

pub fn noisy_exponential_fit_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [0.1, 1.0, 0.0]
  let assert Ok(result) = fit_to_curve(x, exponential, params, trr, noisy: True)
  are_fits_equivalent(x, exponential, params, result)
  |> should.be_true
}

pub fn noisy_parabola_fit_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [-1.4, 0.1, 0.4]
  let assert Ok(result) = fit_to_curve(x, parabola, params, trr, noisy: True)
  are_fits_equivalent(x, parabola, params, result) |> should.be_true
}

pub fn noisy_gaussian_fit_test() {
  let x = generate_x_axis(-50, 50, 100)
  let params = [0.1, 10.0]
  let assert Ok(result) = fit_to_curve(x, gaussian, params, trr, noisy: True)
  are_fits_equivalent(x, gaussian, params, result) |> should.be_true
}

pub fn noisy_double_gaussian_fit_test() {
  let x = generate_x_axis(-3, 7, 100)
  let params = [1.2, 0.3, 0.5, 2.5, 2.0, 1.0]
  let assert Ok(result) =
    fit_to_curve(x, double_gaussian, params, trr, noisy: True)
  are_fits_equivalent(x, double_gaussian, params, result)
  |> should.be_true
}

pub fn noisy_triple_gaussian_fit_test() {
  let x = generate_x_axis(-3, 7, 100)
  let params = [1.2, 0.3, 0.5, 2.5, 2.0, 1.0, 1.0, -2.0, 0.1]
  let assert Ok(result) =
    fit_to_curve(x, triple_gaussian, params, trr, noisy: True)
  are_fits_equivalent(x, triple_gaussian, params, result)
  |> should.be_true
}

pub fn should_error_when_x_y_different_sizes_test() {
  trr([0.0], [], parabola, []) |> should.be_error
}

pub fn should_error_when_lower_bound_diff_size_params_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 0.0, 0.0]
  let y = list.map(x, parabola(_, params))
  gleastsq.trust_region_reflective(
    x,
    y,
    parabola,
    [1.0, 1.0, 1.0],
    lower_bounds: Some([0.0, 0.0]),
    upper_bounds: Some([9.0, 9.0, 9.0]),
    opts: [],
  )
  |> should.be_error
}

pub fn should_error_when_upper_bound_diff_size_params_test() {
  let x = generate_x_axis(-5, 5, 100)
  let params = [2.0, 0.0, 0.0]
  let y = list.map(x, parabola(_, params))
  gleastsq.trust_region_reflective(
    x,
    y,
    parabola,
    [1.0, 1.0, 1.0],
    lower_bounds: Some([0.0, 0.0, 0.0]),
    upper_bounds: Some([9.0, 9.0]),
    opts: [],
  )
  |> should.be_error
}
