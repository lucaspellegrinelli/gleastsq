import gleam/list
import gleam_community/maths/metrics.{standard_deviation}
import prng/random
import prng/seed
import utils/math_utils.{gaussian}

pub fn sample_around(
  x: List(Float),
  f: fn(Float, List(Float)) -> Float,
  params: List(Float),
) -> List(Float) {
  let seed = seed.new(0)
  let y = list.map(x, f(_, params))
  let assert Ok(ampl) = standard_deviation(y, 0)
  let noise_gen = {
    use x <- random.then(random.float(-3.14, 3.14))
    use sign <- random.then(random.choose(-1.0, 1.0))
    random.constant(sign *. gaussian(x, [0.0, 1.0]) *. ampl)
  }

  noise_gen
  |> random.fixed_size_list(list.length(x))
  |> random.sample(seed)
  |> list.zip(y)
  |> list.map(fn(a) { a.0 +. a.1 })
}
