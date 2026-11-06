"""Conservative telemetry-only event candidates for the Week 2 feasibility pilot."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from eventground_wam.demonstration_contracts import (
    EventCandidate,
    TrajectoryRecord,
)
from eventground_wam.demonstration_reader import EpisodeRecord, TrajectoryRow

LIBERO_OPEN_COMMAND = -1.0
LIBERO_CLOSE_COMMAND = 1.0
GRIPPER_CLOSE_DETECTOR_NAME = "gripper-close-telemetry"
GRIPPER_CLOSE_DETECTOR_VERSION = "v1"


class EventLabelingError(ValueError):
    """Raised when pilot inputs cannot support deterministic event extraction."""


@dataclass(frozen=True)
class GripperCloseDetectorConfig:
    """Locked pilot thresholds for an action-plus-proprioception close event."""

    response_window_frames: int = 5
    minimum_close_command_frames: int = 5
    minimum_aperture_drop: float = 0.001
    candidate_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.response_window_frames < 1:
            raise ValueError("response_window_frames must be positive")
        if self.minimum_close_command_frames < self.response_window_frames:
            raise ValueError(
                "minimum_close_command_frames must cover response_window_frames"
            )
        if not math.isfinite(self.minimum_aperture_drop) or self.minimum_aperture_drop <= 0:
            raise ValueError("minimum_aperture_drop must be finite and positive")
        if not math.isfinite(self.candidate_confidence) or not 0 <= self.candidate_confidence <= 1:
            raise ValueError("candidate_confidence must be between zero and one")


@dataclass(frozen=True)
class CandidateExtractionSummary:
    """Counts from one episode's mutually exclusive detector outcomes."""

    close_transition_count: int
    candidate_count: int
    rejected_short_command: int
    rejected_incomplete_response: int
    rejected_weak_aperture_response: int

    @property
    def rejected_count(self) -> int:
        return self.close_transition_count - self.candidate_count


@dataclass(frozen=True)
class CandidateExtractionResult:
    """Raw candidate contracts and detector diagnostics for one episode."""

    candidates: tuple[EventCandidate, ...]
    summary: CandidateExtractionSummary


@dataclass
class GripperPolarityAudit:
    """Aggregate transition evidence for the open/close command convention."""

    response_window_frames: int = 5
    response_threshold: float = 0.0001
    open_transition_count: int = 0
    open_aperture_increase_count: int = 0
    close_transition_count: int = 0
    close_aperture_decrease_count: int = 0

    def update(self, rows: Sequence[TrajectoryRow]) -> None:
        _validate_rows(rows)
        apertures = [_gripper_aperture(row) for row in rows]
        commands = [_gripper_command(row) for row in rows]
        for frame_index in range(1, len(rows)):
            command = commands[frame_index]
            if command == commands[frame_index - 1]:
                continue
            response_index = frame_index + self.response_window_frames
            if response_index >= len(rows):
                continue
            response = apertures[response_index] - apertures[frame_index]
            if command == LIBERO_OPEN_COMMAND:
                self.open_transition_count += 1
                self.open_aperture_increase_count += response >= self.response_threshold
            else:
                self.close_transition_count += 1
                self.close_aperture_decrease_count += response <= -self.response_threshold

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "response_window_frames": self.response_window_frames,
            "response_threshold": self.response_threshold,
            "open_command": LIBERO_OPEN_COMMAND,
            "close_command": LIBERO_CLOSE_COMMAND,
            "open_transition_count": self.open_transition_count,
            "open_aperture_increase_count": self.open_aperture_increase_count,
            "open_response_rate": _ratio(
                self.open_aperture_increase_count,
                self.open_transition_count,
            ),
            "close_transition_count": self.close_transition_count,
            "close_aperture_decrease_count": self.close_aperture_decrease_count,
            "close_response_rate": _ratio(
                self.close_aperture_decrease_count,
                self.close_transition_count,
            ),
        }


def select_task_stratified_episodes(
    episodes: Iterable[EpisodeRecord],
    *,
    episodes_per_task: int = 1,
) -> tuple[EpisodeRecord, ...]:
    """Select deterministic length-stratified episodes within every task."""

    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    by_task: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_index].append(episode)
    if not by_task:
        raise EventLabelingError("cannot select a pilot from an empty episode stream")

    selected: list[EpisodeRecord] = []
    for task_index in sorted(by_task):
        task_episodes = sorted(
            by_task[task_index],
            key=lambda episode: (episode.length, episode.episode_index),
        )
        if episodes_per_task > len(task_episodes):
            raise EventLabelingError(
                f"task {task_index} has only {len(task_episodes)} episodes, "
                f"cannot select {episodes_per_task}"
            )
        positions = _interior_positions(len(task_episodes), episodes_per_task)
        selected.extend(task_episodes[position] for position in positions)
    return tuple(sorted(selected, key=lambda episode: episode.episode_index))


def extract_gripper_close_candidates(
    rows: Sequence[TrajectoryRow],
    trajectory: TrajectoryRecord,
    *,
    config: GripperCloseDetectorConfig | None = None,
) -> CandidateExtractionResult:
    """Detect physical gripper closing after a persistent LIBERO close command."""

    detector_config = config or GripperCloseDetectorConfig()
    _validate_rows(rows, trajectory=trajectory)
    apertures = [_gripper_aperture(row) for row in rows]
    commands = [_gripper_command(row) for row in rows]
    candidates: list[EventCandidate] = []
    transitions = 0
    rejected_short = 0
    rejected_incomplete = 0
    rejected_weak = 0

    for frame_index in range(1, len(rows)):
        if not (
            commands[frame_index - 1] == LIBERO_OPEN_COMMAND
            and commands[frame_index] == LIBERO_CLOSE_COMMAND
        ):
            continue
        transitions += 1
        run_end = frame_index + 1
        while run_end < len(rows) and commands[run_end] == LIBERO_CLOSE_COMMAND:
            run_end += 1
        command_frames = run_end - frame_index
        if command_frames < detector_config.minimum_close_command_frames:
            rejected_short += 1
            continue

        response_index = frame_index + detector_config.response_window_frames
        if response_index >= len(rows):
            rejected_incomplete += 1
            continue
        aperture_drop = apertures[frame_index] - apertures[response_index]
        if aperture_drop < detector_config.minimum_aperture_drop:
            rejected_weak += 1
            continue

        candidates.append(
            EventCandidate(
                provenance=trajectory.provenance,
                trajectory_digest=trajectory.digest,
                frame_start=frame_index,
                frame_end=response_index + 1,
                event_type="gripper_close",
                confidence=detector_config.candidate_confidence,
                evidence=(
                    (
                        "action:LIBERO -1=open to +1=close command transition; "
                        f"+1 persisted for {command_frames} frames"
                    ),
                    (
                        "state:two-finger aperture abs(q0)+abs(q1) dropped "
                        f"{aperture_drop:.8f} over "
                        f"{detector_config.response_window_frames} frames"
                    ),
                ),
                detector_name=GRIPPER_CLOSE_DETECTOR_NAME,
                detector_version=GRIPPER_CLOSE_DETECTOR_VERSION,
            )
        )

    return CandidateExtractionResult(
        candidates=tuple(candidates),
        summary=CandidateExtractionSummary(
            close_transition_count=transitions,
            candidate_count=len(candidates),
            rejected_short_command=rejected_short,
            rejected_incomplete_response=rejected_incomplete,
            rejected_weak_aperture_response=rejected_weak,
        ),
    )


def review_item(
    candidate: EventCandidate,
    trajectory: TrajectoryRecord,
    *,
    context_frames: int = 10,
) -> dict[str, object]:
    """Build a compact, source-referenced item for later human boundary review."""

    if context_frames < 0:
        raise ValueError("context_frames cannot be negative")
    start = max(0, candidate.frame_start - context_frames)
    end = min(trajectory.frame_end, candidate.frame_end + context_frames)
    return {
        "candidate_digest": candidate.digest,
        "trajectory_digest": trajectory.digest,
        "source_episode_index": trajectory.provenance.source_episode_index,
        "source_task_index": trajectory.provenance.source_task_index,
        "instruction": trajectory.instruction,
        "event_type": candidate.event_type,
        "candidate_frame_range": [candidate.frame_start, candidate.frame_end],
        "review_frame_range": [start, end],
        "candidate_time_seconds": [
            candidate.frame_start / trajectory.fps,
            (candidate.frame_end - 1) / trajectory.fps,
        ],
        "front_video_path": trajectory.front_video_path,
        "wrist_video_path": trajectory.wrist_video_path,
        "review_status": "pending",
    }


def _validate_rows(
    rows: Sequence[TrajectoryRow],
    *,
    trajectory: TrajectoryRecord | None = None,
) -> None:
    if not rows:
        raise EventLabelingError("event extraction requires a nonempty episode")
    expected_episode = rows[0].episode_index
    expected_task = rows[0].task_index
    for frame_index, row in enumerate(rows):
        if row.frame_index != frame_index:
            raise EventLabelingError("trajectory rows are not in contiguous frame order")
        if row.episode_index != expected_episode or row.task_index != expected_task:
            raise EventLabelingError("trajectory rows cross an episode or task boundary")
    if trajectory is not None:
        provenance = trajectory.provenance
        if (
            provenance.source_episode_index != expected_episode
            or provenance.source_task_index != expected_task
            or provenance.source_episode_length != len(rows)
        ):
            raise EventLabelingError("trajectory contract does not identify the supplied rows")


def _gripper_aperture(row: TrajectoryRow) -> float:
    return abs(row.state[-2]) + abs(row.state[-1])


def _gripper_command(row: TrajectoryRow) -> float:
    command = row.action[-1]
    if command not in {LIBERO_OPEN_COMMAND, LIBERO_CLOSE_COMMAND}:
        raise EventLabelingError(
            f"episode {row.episode_index} frame {row.frame_index} has "
            f"unsupported gripper command {command}"
        )
    return command


def _interior_positions(total: int, count: int) -> tuple[int, ...]:
    if count == 1:
        return ((total - 1) // 2,)
    positions = tuple(round(index * (total - 1) / (count - 1)) for index in range(count))
    if len(set(positions)) != count:
        raise EventLabelingError("length strata collapsed to duplicate episode positions")
    return positions


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
