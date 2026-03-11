-module(gleastsq_jacobian_ffi).
-export([parallel_map/2]).

parallel_map(Items, Fun) ->
    Parent = self(),
    Tag = make_ref(),
    Count = length(Items),
    _ = [
        spawn(fun() ->
            Result =
                try
                    {ok, Fun(Item)}
                catch
                    _:_ ->
                        {error, nil}
                end,
            Parent ! {Tag, Index, Result}
        end)
     || {Index, Item} <- lists:zip(lists:seq(1, Count), Items)
    ],
    gather_results(Tag, Count, #{}).

gather_results(_Tag, 0, Results) ->
    {ok, [maps:get(Index, Results) || Index <- lists:seq(1, maps:size(Results))]};
gather_results(Tag, Remaining, Results) ->
    receive
        {Tag, Index, {ok, Value}} ->
            gather_results(Tag, Remaining - 1, maps:put(Index, Value, Results));
        {Tag, _Index, {error, nil}} ->
            {error, nil}
    end.
