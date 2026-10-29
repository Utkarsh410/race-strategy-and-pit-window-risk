from racesim.schemas import Strategy
from racesim.strategy_generator import (
    generate_candidate_strategies, generate_compound_sequences, generate_stop_sequences, remove_duplicate_strategies,
    validate_strategy,
)


def test_stop_sequences_respect_stint_limits():
    seqs = generate_stop_sequences(1, 30, 2, min_stint_laps=5, max_stint_laps=15, min_laps_remaining=1)
    assert seqs
    for s in seqs:
        bounds = [0, *s, 30]
        lengths = [b - a for a, b in zip(bounds, bounds[1:])]
        assert all(5 <= x <= 15 for x in lengths)


def test_compound_rule(simple_setup):
    seqs = generate_compound_sequences(simple_setup, 1)
    assert ("FAST", "FAST") not in seqs and ("FAST", "SLOW") in seqs


def test_candidates_valid_deterministic_and_limited(make_simple_setup):
    s = make_simple_setup(maximum_strategy_candidates=60)
    a = generate_candidate_strategies(s)
    b = generate_candidate_strategies(s)
    assert [x.strategy_id for x in a.strategies] == [x.strategy_id for x in b.strategies]
    assert 0 < len(a) <= 60
    assert all(validate_strategy(x, s).valid for x in a.strategies)
    assert any("coarsened" in r for _, r in a.rejected)
    assert all(x.number_of_stops >= 1 for x in a.strategies)  # mandatory stop


def test_validation_reasons(simple_setup):
    v = validate_strategy(Strategy.from_plan(20, [], ["FAST"]), simple_setup)
    assert not v.valid
    assert any("mandatory" in r for r in v.reasons) and any("distinct" in r for r in v.reasons)
    v2 = validate_strategy(Strategy.from_plan(20, [1], ["FAST", "SLOW"]), simple_setup)
    assert any("minimum" in r for r in v2.reasons)


def test_remove_duplicates():
    s = Strategy.from_plan(20, [10], ["FAST", "SLOW"])
    assert len(remove_duplicate_strategies([s, s, Strategy.from_plan(20, [11], ["FAST", "SLOW"])])) == 2
