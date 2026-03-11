import gleam/float
import gleam/list
import gleam/option.{None, Some}
import gleastsq
import gleastsq/errors.{SolveError}
import gleastsq/internal/methods/trust_region_reflective as trr_impl
import gleastsq/internal/nx
import gleastsq/options.{Damping}
import gleeunit/should
import utils/curves.{
  double_gaussian, exponential, gaussian, parabola, triple_gaussian,
}
import utils/helpers.{
  are_fits_equivalent, fit_to_curve, generate_x_axis, sum_squared_residuals,
}

pub fn trr(
  x: List(Float),
  y: List(Float),
  f: fn(Float, List(Float)) -> Float,
  p: List(Float),
) {
  gleastsq.trust_region_reflective(x, y, f, p, None, None, [])
}

fn linear(x: Float, params: List(Float)) -> Float {
  let assert [a] = params
  a *. x
}

fn redundant_constant(_x: Float, params: List(Float)) -> Float {
  let assert [a, b] = params
  a +. b
}

fn params_within_bounds(
  params: List(Float),
  lower_bounds: List(Float),
  upper_bounds: List(Float),
) -> Bool {
  list.zip(params, lower_bounds)
  |> list.zip(upper_bounds)
  |> list.all(fn(pair) {
    let #(param_and_lower, upper) = pair
    let #(param, lower) = param_and_lower
    param >=. lower && param <=. upper
  })
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

pub fn should_error_when_initial_params_empty_test() {
  trr([0.0], [0.0], parabola, []) |> should.be_error
}

pub fn rho_uses_quadratic_model_test() {
  let x = [1.0, 2.0]
  let y = [1.0, 2.0] |> nx.tensor
  let params = [0.0]
  let p = [0.5] |> nx.tensor
  let g = [-5.0] |> nx.tensor
  let b = [[5.0]] |> nx.tensor

  trr_impl.rho(x, y, linear, params, p, g, b)
  |> float.loosely_equals(1.0, tolerating: 0.0000001)
  |> should.equal(True)
}

pub fn rho_returns_zero_for_zero_predicted_reduction_test() {
  let x = [1.0]
  let y = [1.0] |> nx.tensor
  let params = [0.0]
  let p = [0.0] |> nx.tensor
  let g = [0.0] |> nx.tensor
  let b = [[1.0]] |> nx.tensor

  trr_impl.rho(x, y, linear, params, p, g, b)
  |> should.equal(0.0)
}

pub fn fixed_bounds_zero_step_returns_current_params_test() {
  let x = [1.0, 2.0]
  let y = [1.0, 2.0]
  gleastsq.trust_region_reflective(
    x,
    y,
    linear,
    [0.0],
    lower_bounds: Some([0.0]),
    upper_bounds: Some([0.0]),
    opts: [],
  )
  |> should.equal(Ok([0.0]))
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

pub fn successful_fit_reduces_residual_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [0.1, 1.0, 0.0]
  let y = list.map(x, exponential(_, params))
  let initial = [1.0, 1.0, 1.0]
  let initial_residual = sum_squared_residuals(x, y, exponential, initial)
  let assert Ok(result) =
    gleastsq.trust_region_reflective(
      x,
      y,
      exponential,
      initial,
      lower_bounds: None,
      upper_bounds: None,
      opts: [],
    )
  let final_residual = sum_squared_residuals(x, y, exponential, result)
  should.be_true(final_residual <. initial_residual)
}

pub fn should_return_solve_error_for_rank_deficient_problem_without_damping_test() {
  let result =
    gleastsq.trust_region_reflective(
      [0.0, 1.0],
      [1.0, 2.0],
      redundant_constant,
      [0.0, 0.0],
      lower_bounds: None,
      upper_bounds: None,
      opts: [Damping(0.0)],
    )

  case result {
    Error(SolveError(_)) -> should.be_true(True)
    _ -> should.be_true(False)
  }
}

pub fn result_params_stay_within_bounds_test() {
  let x = generate_x_axis(0, 5, 100)
  let params = [0.6, 0.8, 0.2]
  let y = list.map(x, exponential(_, params))
  let lower_bounds = [0.0, 0.0, -0.5]
  let upper_bounds = [0.7, 1.0, 0.4]
  let assert Ok(result) =
    gleastsq.trust_region_reflective(
      x,
      y,
      exponential,
      [10.0, -10.0, 5.0],
      lower_bounds: Some(lower_bounds),
      upper_bounds: Some(upper_bounds),
      opts: [],
    )

  params_within_bounds(result, lower_bounds, upper_bounds)
  |> should.be_true
}

pub fn initial_params_are_clipped_to_fixed_bounds_before_optimization_test() {
  gleastsq.trust_region_reflective(
    [1.0, 2.0],
    [1.0, 2.0],
    linear,
    [10.0],
    lower_bounds: Some([2.0]),
    upper_bounds: Some([2.0]),
    opts: [],
  )
  |> should.equal(Ok([2.0]))
}

pub fn solution_can_land_on_active_upper_bound_test() {
  let assert Ok(result) =
    gleastsq.trust_region_reflective(
      [1.0, 2.0, 3.0],
      [3.0, 6.0, 9.0],
      linear,
      [0.0],
      lower_bounds: Some([0.0]),
      upper_bounds: Some([2.0]),
      opts: [],
    )

  result
  |> should.equal([2.0])
}
