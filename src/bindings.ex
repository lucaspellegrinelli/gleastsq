defmodule NxBindings do
  require Nx

  def concatenate(tensor_list, axis) do
    Nx.concatenate(tensor_list, [{:axis, axis}])
  end
end
