import gleam/list
import gleam_community/maths.{standard_deviation}
import prng/random
import utils/curves.{gaussian}

pub fn sample_around(
  x: List(Float),
  f: fn(Float, List(Float)) -> Float,
  params: List(Float),
) -> List(Float) {
  let seed = random.new_seed(0)
  let y = list.map(x, f(_, params))
  let assert Ok(ampl) = standard_deviation(y, 0)
  let noise_gen = {
    use x <- random.then(random.float(-3.14, 3.14))
    use sign <- random.then(random.choose(-1.0, 1.0))
    random.constant(sign *. gaussian(x, [0.0, 1.0]) *. ampl)
  }
  let #(noise, _) =
    random.fixed_size_list(from: noise_gen, of: list.length(x))
    |> random.step(seed)

  noise
  |> list.zip(y)
  |> list.map(fn(a) { a.0 +. a.1 })
}
