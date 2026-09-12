from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from eventground_wam.b0_data import (
    ActionNormalizer,
    B0DataConfig,
    B0DataError,
    B0DataLoader,
    B0SplitIndex,
    B0StreamingDataset,
    fit_b0_action_statistics,
    validate_b0_action_roundtrip,
)
from eventground_wam.demonstration_contracts import (
    Provenance,
    SplitAssignment,
    contract_digest,
)
from eventground_wam.demonstration_reader import (
    DecodedVideoFrame,
    EpisodeRecord,
    SynchronizedStep,
    TrajectoryRow,
)

REVISION = "1" * 40


class FakeReader:
    def __init__(
        self,
        lengths: list[int],
        *,
        continuous_offset_by_episode: dict[int, float] | None = None,
    ) -> None:
        self.row_opens: list[int] = []
        self.step_opens: list[int] = []
        self._offsets = continuous_offset_by_episode or {}
        start_index = 0
        episodes: list[EpisodeRecord] = []
        for episode_index, length in enumerate(lengths):
            episodes.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    task_index=episode_index % 2,
                    instruction=f"instruction {episode_index}",
                    length=length,
                    start_index=start_index,
                    parquet_path=Path(f"episode-{episode_index}.parquet"),
                    video_paths=(
                        ("observation.images.front", Path(f"front-{episode_index}.mp4")),
                        ("observation.images.wrist", Path(f"wrist-{episode_index}.mp4")),
                    ),
                )
            )
            start_index += length
        self._episodes = tuple(episodes)

    def iter_episodes(self) -> Iterator[EpisodeRecord]:
        yield from self._episodes

    def iter_episode_rows(self, episode: EpisodeRecord) -> Iterator[TrajectoryRow]:
        self.row_opens.append(episode.episode_index)
        for frame_index in range(episode.length):
            yield self._row(episode, frame_index)

    def iter_episode_steps(self, episode: EpisodeRecord) -> Iterator[SynchronizedStep]:
        self.step_opens.append(episode.episode_index)
        for frame_index in range(episode.length):
            row = self._row(episode, frame_index)
            yield SynchronizedStep(
                row=row,
                front_frame=self._frame(episode.episode_index, frame_index, camera=0),
                wrist_frame=self._frame(episode.episode_index, frame_index, camera=1),
            )

    def _row(self, episode: EpisodeRecord, frame_index: int) -> TrajectoryRow:
        offset = self._offsets.get(episode.episode_index, episode.episode_index * 100.0)
        continuous = tuple(offset + frame_index + dimension / 10.0 for dimension in range(6))
        gripper = -1.0 if frame_index % 2 == 0 else 1.0
        return TrajectoryRow(
            index=episode.start_index + frame_index,
            episode_index=episode.episode_index,
            frame_index=frame_index,
            task_index=episode.task_index,
            timestamp=frame_index / 20.0,
            state=tuple(
                episode.episode_index * 1_000.0 + frame_index * 10.0 + dimension
                for dimension in range(8)
            ),
            action=(*continuous, gripper),
        )

    @staticmethod
    def _frame(episode_index: int, frame_index: int, *, camera: int) -> DecodedVideoFrame:
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        image[0, 0, 0] = episode_index
        image[1, 0, 0] = frame_index
        image[2, 0, 0] = 10 + camera
        image[:, -1, -1] = np.array((20 + camera, 30 + camera, 40 + camera))
        return DecodedVideoFrame(
            frame_index=frame_index,
            timestamp=frame_index / 20.0,
            width=256,
            height=256,
            pixel_format="yuv420p",
            payload=image,
        )


def _split_index(lengths: list[int], splits: list[str]) -> B0SplitIndex:
    records = []
    for episode_index, (length, split) in enumerate(zip(lengths, splits, strict=True)):
        records.append(
            SplitAssignment(
                provenance=Provenance(
                    source_episode_index=episode_index,
                    source_task_index=episode_index % 2,
                    source_episode_length=length,
                    project_git_revision=REVISION,
                    annotation_version="test-splits-v1",
                ),
                trajectory_digest=f"{episode_index + 1:064x}",
                frame_start=0,
                frame_end=length,
                split=split,
                group_strategy="task-state-rgb-surrogate-v1",
                initial_condition_digest=f"{episode_index + 100:064x}",
            ).to_dict()
        )
    return B0SplitIndex.from_records(records, expected_episode_count=len(lengths))


def _dataset(
    lengths: list[int],
    splits: list[str],
    *,
    config: B0DataConfig | None = None,
    reader: FakeReader | None = None,
) -> tuple[FakeReader, B0StreamingDataset]:
    fake_reader = reader or FakeReader(lengths)
    dataset = B0StreamingDataset(
        fake_reader,  # type: ignore[arg-type]
        _split_index(lengths, splits),
        ActionNormalizer.identity(),
        config=config or B0DataConfig(shuffle_episodes=False),
        rgb_frame_converter=lambda frame: frame.payload,
    )
    return fake_reader, dataset


@pytest.mark.parametrize(("length", "expected"), [(7, 0), (8, 1), (9, 2)])
def test_incomplete_tail_chunks_are_excluded(length: int, expected: int) -> None:
    _, dataset = _dataset([length], ["train"])

    assert dataset.sample_count == expected
    assert len(list(dataset.iter_samples())) == expected


def test_sample_aligns_anchor_modalities_and_eight_actions_without_crossing() -> None:
    _, dataset = _dataset([8, 9], ["train", "train"])

    samples = list(dataset.iter_samples())

    assert [(sample.identity.episode_index, sample.identity.anchor_frame_index) for sample in samples] == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    second = samples[1]
    assert second.instruction == "instruction 1"
    assert second.front_rgb[:, 0, 0].tolist() == [1, 0, 10]
    assert second.wrist_rgb[:, 0, 0].tolist() == [1, 0, 11]
    assert second.proprioception.tolist() == pytest.approx(
        [1_000.0 + dimension for dimension in range(8)]
    )
    assert second.actions.shape == (8, 7)
    assert second.actions[:, 0].tolist() == pytest.approx([100.0 + index for index in range(8)])
    assert second.actions[:, 6].tolist() == [-1.0, 1.0] * 4
    assert all(
        np.all((sample.actions[:, 0] < 50.0) if sample.identity.episode_index == 0 else (sample.actions[:, 0] > 50.0))
        for sample in samples
    )


def test_training_stream_never_opens_validation_or_test_payloads_and_is_lazy() -> None:
    reader, dataset = _dataset([8, 8, 8], ["train", "validation", "test"])
    assert reader.step_opens == []
    assert reader.row_opens == []

    samples = list(dataset.iter_samples())

    assert len(samples) == 1
    assert samples[0].identity.episode_index == 0
    assert reader.step_opens == [0]
    assert reader.row_opens == []


def test_identity_order_shuffle_repeatability_and_episode_sharding() -> None:
    _, canonical = _dataset(
        [8, 9, 10, 11],
        ["train"] * 4,
        config=B0DataConfig(seed=17, shuffle_episodes=False),
    )
    canonical_ids = list(canonical.iter_sample_identities())
    assert [(item.episode_index, item.anchor_frame_index) for item in canonical_ids] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ]

    _, shuffled = _dataset(
        [8, 9, 10, 11],
        ["train"] * 4,
        config=B0DataConfig(seed=17, shuffle_episodes=True),
    )
    first = list(shuffled.iter_sample_identities(epoch=2))
    second = list(shuffled.iter_sample_identities(epoch=2))
    other_seed_reader = FakeReader([8, 9, 10, 11])
    other_seed = B0StreamingDataset(
        other_seed_reader,  # type: ignore[arg-type]
        _split_index([8, 9, 10, 11], ["train"] * 4),
        ActionNormalizer.identity(),
        config=B0DataConfig(seed=23),
        rgb_frame_converter=lambda frame: frame.payload,
    )
    assert [item.digest for item in first] == [item.digest for item in second]
    assert shuffled.ordered_identity_digest(epoch=2) == shuffled.ordered_identity_digest(epoch=2)
    assert [item.digest for item in first] != [
        item.digest for item in other_seed.iter_sample_identities(epoch=2)
    ]
    assert {item.digest for item in first} == {item.digest for item in canonical_ids}

    shards = [
        list(shuffled.iter_sample_identities(epoch=2, shard_index=index, shard_count=2))
        for index in range(2)
    ]
    assert len(shards[0]) + len(shards[1]) == len(first)
    assert {item.digest for shard in shards for item in shard} == {item.digest for item in first}
    assert not ({item.digest for item in shards[0]} & {item.digest for item in shards[1]})


def test_explicit_rgb_orientation_and_batch_shapes() -> None:
    _, identity_dataset = _dataset([9], ["train"])
    identity_batch = next(B0DataLoader(identity_dataset, batch_size=2).iter_batches())

    assert identity_batch.front_rgb.shape == (2, 3, 256, 256)
    assert identity_batch.wrist_rgb.shape == (2, 3, 256, 256)
    assert identity_batch.proprioception.shape == (2, 8)
    assert identity_batch.actions.shape == (2, 8, 7)
    assert identity_batch.front_rgb.dtype == np.uint8
    assert identity_batch.proprioception.dtype == np.float32
    assert identity_batch.actions.dtype == np.float32
    assert identity_batch.front_rgb[0, :, 0, 0].tolist() == [0, 0, 10]

    _, rotated_dataset = _dataset(
        [8],
        ["train"],
        config=B0DataConfig(shuffle_episodes=False, camera_orientation="rotate-180"),
    )
    rotated = next(rotated_dataset.iter_samples())
    assert rotated.front_rgb[:, 0, 0].tolist() == [20, 30, 40]
    assert rotated.wrist_rgb[:, 0, 0].tolist() == [21, 31, 41]
    assert rotated.front_rgb.flags.c_contiguous


def test_train_only_statistics_count_rows_once_and_never_decode_video() -> None:
    lengths = [8, 9, 8]
    splits = ["train", "train", "validation"]
    reader = FakeReader(
        lengths,
        continuous_offset_by_episode={0: 0.0, 1: 20.0, 2: 10_000.0},
    )
    index = _split_index(lengths, splits)

    artifact = fit_b0_action_statistics(
        reader,  # type: ignore[arg-type]
        index,
        project_git_revision=REVISION,
        project_tree_clean=True,
    )

    assert artifact["split"] == {
        "name": "train",
        "episode_count": 2,
        "action_count": 17,
        "valid_action_chunk_count": 3,
        "action_chunk_length": 8,
    }
    assert artifact["statistics"]["raw_min"][:6] == pytest.approx(
        [dimension / 10.0 for dimension in range(6)]
    )
    assert artifact["statistics"]["raw_max"][:6] == pytest.approx(
        [28.0 + dimension / 10.0 for dimension in range(6)]
    )
    assert max(artifact["statistics"]["upper_q99"][:6]) < 100.0
    assert artifact["validation"]["actions_counted_once_not_from_overlapping_chunks"] is True
    assert reader.row_opens == [0, 1]
    assert reader.step_opens == []

    normalizer = ActionNormalizer.from_artifact(artifact)
    actions = np.asarray(
        [
            [-20.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0],
            [30.0, -2.0, 1.0, -4.0, 2.0, 0.0, 1.0],
        ]
    )
    reconstructed = normalizer.denormalize(normalizer.normalize(actions))
    np.testing.assert_allclose(reconstructed, actions, rtol=0.0, atol=1e-6)
    assert normalizer.normalize(actions)[:, 6].tolist() == [-1.0, 1.0]
    assert reconstructed[:, 6].tolist() == [-1.0, 1.0]

    validation_reader = FakeReader(lengths)
    validation = validate_b0_action_roundtrip(
        validation_reader,  # type: ignore[arg-type]
        index,
        normalizer,
    )
    assert validation["action_counts_by_split"] == {
        "train": 17,
        "validation": 8,
        "test": 0,
    }
    assert validation["gripper_polarity_preserved"] is True
    assert validation_reader.step_opens == []


def test_constant_continuous_dimension_uses_finite_unit_scale() -> None:
    reader = FakeReader([8], continuous_offset_by_episode={0: 3.0})

    artifact = fit_b0_action_statistics(
        reader,  # type: ignore[arg-type]
        _split_index([8], ["train"]),
        project_git_revision=REVISION,
        project_tree_clean=True,
    )
    # Values change by frame in the fixture, so force the serialized constant policy directly.
    artifact["statistics"]["lower_q01"][0] = 3.0
    artifact["statistics"]["upper_q99"][0] = 3.0
    artifact["statistics"]["center"][0] = 3.0
    artifact["statistics"]["scale"][0] = 1.0
    artifact["digest"] = contract_digest(artifact)
    normalizer = ActionNormalizer.from_artifact(artifact)

    assert normalizer.scale[0] == 1.0
    assert np.isfinite(normalizer.normalize([[3.0, 0, 0, 0, 0, 0, -1]])).all()


def test_split_assignments_must_be_complete_unique_and_match_reader() -> None:
    index = _split_index([8, 8], ["train", "validation"])
    reader = FakeReader([8, 9])

    with pytest.raises(B0DataError, match="metadata differs"):
        B0StreamingDataset(
            reader,  # type: ignore[arg-type]
            index,
            ActionNormalizer.identity(),
            rgb_frame_converter=lambda frame: frame.payload,
        )

    one_record = SplitAssignment(
        provenance=Provenance(
            source_episode_index=1,
            source_task_index=1,
            source_episode_length=8,
            project_git_revision=REVISION,
            annotation_version="test-splits-v1",
        ),
        trajectory_digest="1" * 64,
        frame_start=0,
        frame_end=8,
        split="train",
        group_strategy="task-state-rgb-surrogate-v1",
        initial_condition_digest="2" * 64,
    ).to_dict()
    with pytest.raises(B0DataError, match="expected episode 0"):
        B0SplitIndex.from_records([one_record])


def test_normalizer_rejects_mismatched_frozen_source() -> None:
    normalizer = ActionNormalizer(
        center=(0.0,) * 7,
        scale=(1.0,) * 7,
        lower=(-1.0,) * 7,
        upper=(1.0,) * 7,
        source_identity={"dataset_version": "different"},
    )
    with pytest.raises(B0DataError, match="different frozen sources"):
        B0StreamingDataset(
            FakeReader([8]),  # type: ignore[arg-type]
            _split_index([8], ["train"]),
            normalizer,
            rgb_frame_converter=lambda frame: frame.payload,
        )


def test_committed_real_data_evidence_is_sealed_and_complete() -> None:
    path = Path(__file__).parents[1] / "docs/evaluation/week3-b0-data-v1.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["digest"] == contract_digest(evidence)
    assert evidence["status"] == "passed"
    assert evidence["implementation"]["project_tree_clean"] is True
    assert evidence["normalization"]["episode_count"] == 11_521
    assert evidence["normalization"]["action_count"] == 1_796_601
    assert evidence["validation"]["train_sample_count"] == 1_715_954
    assert evidence["validation"]["episode_zero"]["sample_count"] == 270
    assert evidence["validation"]["all_actions_round_trip_passed"] is True
    assert evidence["validation"]["gripper_polarity_preserved"] is True
    assert evidence["validation"]["max_round_trip_absolute_error"] <= 1e-6
    assert evidence["source"]["freeze_manifest_sha256"] == (
        "53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2"
    )
    assert evidence["source"]["split_assignments_sha256"] == (
        "71b8fc2a3ce04c84cd41e3c9f052f1a677a0fa2d546201f8540f57eeb3c46758"
    )
