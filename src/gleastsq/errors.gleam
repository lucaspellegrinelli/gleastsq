pub type FitErrors {
  NonConverged
  WrongParameters(String)
  JacobianTaskError
  SolveError(String)
}
