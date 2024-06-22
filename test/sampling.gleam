import gleam/list
import gleam_community/maths/metrics.{standard_deviation}
import math_utils.{gaussian}
import prng/random
import prng/seed

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
  list.map(y, fn(y) {
    let noise = random.sample(noise_gen, seed)
    y +. noise
  })
}
