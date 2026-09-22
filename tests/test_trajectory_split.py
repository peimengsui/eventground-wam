import numpy as np
import pytest

from eventground_wam.trajectory_split import (
    NumericTrajectory,
    build_trajectory_groups,
    expand_groups,
    split_quotas,
)


def episode(index, value, task="task-00", *, state_offset=0., action_offset=0.):
    states = np.full((9, 8), value + state_offset, dtype=np.float32)
    actions = np.full((9, 7), value + action_offset, dtype=np.float32)
    actions[:, 6] = 1
    return NumericTrajectory(index, task, states, actions)


def test_all_variants_and_near_duplicate_chains_stay_together():
    rows = [episode(i, i / 10) for i in range(10)]
    rows += [episode(10, 0.), episode(11, 0., state_offset=8e-6, action_offset=8e-6),
             episode(12, 0., state_offset=16e-6, action_offset=16e-6)]
    groups, audit = build_trajectory_groups(rows)
    group = next(g for g in groups if 0 in g["episode_indices"])
    assert group["episode_indices"] == [0, 10, 11, 12]
    assert audit["connected_components"] == 10
    assert len(audit["near_edges_between_distinct_exact_sequences"]) == 2
    assignments = {r["episode_index"]: r for r in expand_groups(groups)}
    assert len({assignments[i]["split"] for i in [0, 10, 11, 12]}) == 1


def test_action_only_duplicates_are_conservatively_joined():
    rows = [episode(i, i / 10) for i in range(10)]
    rows.append(episode(10, 0., state_offset=4.))
    groups, _ = build_trajectory_groups(rows)
    assert any(g["episode_indices"] == [0, 10] for g in groups)


def test_assignment_is_order_independent_and_does_not_weight_variants():
    rows = [episode(i, i / 10) for i in range(42)]
    first, _ = build_trajectory_groups(rows)
    repeated, _ = build_trajectory_groups(rows + [episode(99, 0.)])
    shuffled, _ = build_trajectory_groups(list(reversed(rows)))
    assert first == shuffled
    assert [(r["group_digest"], r["split"]) for r in first] == [
        (r["group_digest"], r["split"]) for r in repeated]
    assert split_quotas(42) == {"train": 34, "validation": 4, "test": 4}


def test_stratified_coverage_and_no_cross_split_group_overlap():
    rows = [episode(i, i / 10, f"task-{i // 20:02d}") for i in range(40)]
    groups, _ = build_trajectory_groups(rows)
    for task in {r.task_key for r in rows}:
        assert {split: sum(g["task_key"] == task and g["split"] == split for g in groups)
                for split in ["train", "validation", "test"]} == {
                    "train": 16, "validation": 2, "test": 2}
    assert len(expand_groups(groups)) == len(rows)


def test_reject_cross_task_duplicates_and_insufficient_groups():
    rows = [episode(i, i / 10) for i in range(10)]
    with pytest.raises(ValueError, match="crosses tasks"):
        build_trajectory_groups(rows + [episode(10, 0., "another")])
    with pytest.raises(ValueError, match="at least ten"):
        build_trajectory_groups(rows[:9])
    with pytest.raises(ValueError, match="unique episode"):
        build_trajectory_groups(rows + [rows[0]])


def test_bad_arrays_and_duplicate_assignments_fail():
    a = episode(0, 0.)
    with pytest.raises(ValueError, match="float32"):
        NumericTrajectory(0, "task", a.states.astype(np.float64), a.actions)
    a.states[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        NumericTrajectory(0, "task", a.states, a.actions)
    rows = [episode(i, i / 10) for i in range(10)]
    groups, _ = build_trajectory_groups(rows)
    with pytest.raises(ValueError, match="more than once"):
        expand_groups(groups + groups[:1])
