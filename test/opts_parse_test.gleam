import gleam/option.{None, Some}
import gleastsq/internal/helpers/params.{decode_params}
import gleastsq/options.{
  Damping, DampingDecrease, DampingIncrease, Epsilon, Iterations, Tolerance,
}
import gleeunit/should

pub fn empty_opts_test() {
  let opts = []
  let params = decode_params(opts)
  params.iterations |> should.equal(None)
  params.epsilon |> should.equal(None)
  params.tolerance |> should.equal(None)
  params.damping |> should.equal(None)
  params.damping_increase |> should.equal(None)
  params.damping_decrease |> should.equal(None)
}

pub fn iterations_opts_test() {
  let opts = [Iterations(10)]
  let params = decode_params(opts)
  params.iterations |> should.equal(Some(10))
  params.epsilon |> should.equal(None)
}

pub fn epsilon_opts_test() {
  let opts = [Epsilon(0.1)]
  let params = decode_params(opts)
  params.epsilon |> should.equal(Some(0.1))
  params.iterations |> should.equal(None)
}

pub fn tolerance_opts_test() {
  let opts = [Tolerance(0.01)]
  let params = decode_params(opts)
  params.tolerance |> should.equal(Some(0.01))
  params.iterations |> should.equal(None)
}

pub fn damping_opts_test() {
  let opts = [Damping(0.5)]
  let params = decode_params(opts)
  params.damping |> should.equal(Some(0.5))
  params.iterations |> should.equal(None)
}

pub fn damping_increase_opts_test() {
  let opts = [DampingIncrease(0.1)]
  let params = decode_params(opts)
  params.damping_increase |> should.equal(Some(0.1))
  params.iterations |> should.equal(None)
}

pub fn damping_decrease_opts_test() {
  let opts = [DampingDecrease(0.1)]
  let params = decode_params(opts)
  params.damping_decrease |> should.equal(Some(0.1))
  params.iterations |> should.equal(None)
}

pub fn multiple_opts_test() {
  let opts = [
    Iterations(10),
    Epsilon(0.1),
    Tolerance(0.01),
    Damping(0.5),
    DampingIncrease(0.1),
    DampingDecrease(0.1),
  ]
  let params = decode_params(opts)
  params.iterations |> should.equal(Some(10))
  params.epsilon |> should.equal(Some(0.1))
  params.tolerance |> should.equal(Some(0.01))
  params.damping |> should.equal(Some(0.5))
  params.damping_increase |> should.equal(Some(0.1))
  params.damping_decrease |> should.equal(Some(0.1))
}
