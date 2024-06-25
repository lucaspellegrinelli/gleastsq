defmodule NxBindings do
  require Nx

  def safe_solve(a, b) do
    try do
      result = Nx.LinAlg.solve(a, b)
      {:ok, result}
    rescue
      _exception -> {:error, "Failed to solve"}
    end
  end
end
