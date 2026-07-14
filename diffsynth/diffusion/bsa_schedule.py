"""Serializable block-sparse-attention sparsity schedules.

Schedules are expressed in epochs, optimizer steps, or fractions of the full
training run and compiled once to completed-optimizer-step boundaries.  The
module intentionally has no Torch dependency so schedule parsing can also be
used by launch scripts and checkpoint validation.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


_SUPPORTED_SCHEMA_VERSION = 1
_BASES = frozenset({"epoch", "progress", "optimizer_step"})
_REMAINDER = "remainder"
_TARGET = "target"
_ScheduleInput = Union[str, Mapping[str, Any], "BSAScheduleSpec"]


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not a boolean.")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite decimal number, received {value!r}.") from error
    if not result.is_finite():
        raise ValueError(f"{field} must be finite, received {value!r}.")
    return result


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer.") from error
    if result <= 0 or result != value:
        raise ValueError(f"{field} must be a positive integer, received {value!r}.")
    return result


@dataclass(frozen=True)
class BSAStage:
    """One piecewise-constant schedule stage."""

    sparsity: Union[Decimal, str]
    duration: Union[Decimal, str]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        sparsity = self.sparsity
        if isinstance(sparsity, str) and sparsity.strip().lower() == _TARGET:
            object.__setattr__(self, "sparsity", _TARGET)
        else:
            sparsity = _decimal(sparsity, field="stage sparsity")
            if not Decimal("0") <= sparsity < Decimal("1"):
                raise ValueError(f"Stage sparsity must be in [0, 1), received {sparsity}.")
            object.__setattr__(self, "sparsity", sparsity)

        duration = self.duration
        if isinstance(duration, str) and duration.strip().lower() == _REMAINDER:
            object.__setattr__(self, "duration", _REMAINDER)
        else:
            duration = _decimal(duration, field="stage duration")
            if duration <= 0:
                raise ValueError(f"Stage duration must be positive, received {duration}.")
            object.__setattr__(self, "duration", duration)

        if self.name is not None:
            name = str(self.name).strip()
            if not name:
                raise ValueError("Stage name cannot be empty.")
            object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class BSAScheduleSpec:
    """Versioned, serializable BSA sparsity schedule specification."""

    schema_version: int
    basis: str
    stages: Tuple[BSAStage, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or int(self.schema_version) != self.schema_version:
            raise ValueError("BSA schedule schema_version must be an integer.")
        schema_version = int(self.schema_version)
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported BSA schedule schema_version {schema_version}; "
                f"expected {_SUPPORTED_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "schema_version", schema_version)

        basis = str(self.basis).strip().lower()
        if basis not in _BASES:
            raise ValueError(f"BSA schedule basis must be one of {sorted(_BASES)}, received {basis!r}.")
        object.__setattr__(self, "basis", basis)

        stages = tuple(
            stage if isinstance(stage, BSAStage) else _stage_from_mapping(stage, index=index)
            for index, stage in enumerate(self.stages)
        )
        if not stages:
            raise ValueError("BSA schedule must contain at least one stage.")
        remainder_indices = [index for index, stage in enumerate(stages) if stage.duration == _REMAINDER]
        if remainder_indices and remainder_indices != [len(stages) - 1]:
            raise ValueError("Only the final BSA schedule stage may use duration='remainder'.")
        target_indices = [index for index, stage in enumerate(stages) if stage.sparsity == _TARGET]
        if target_indices and target_indices != [len(stages) - 1]:
            raise ValueError("Only the final BSA schedule stage may use sparsity='target'.")
        for previous, current in zip(stages, stages[1:]):
            if current.sparsity != _TARGET and current.sparsity < previous.sparsity:
                raise ValueError("BSA schedule sparsity must be non-decreasing across stages.")
        object.__setattr__(self, "stages", stages)


@dataclass(frozen=True)
class CompiledBSASchedule:
    """A schedule compiled to completed-optimizer-step transition boundaries."""

    spec: BSAScheduleSpec
    transition_steps: Tuple[int, ...]
    sparsities: Tuple[float, ...]
    total_optimizer_steps: int
    steps_per_epoch: Optional[int]
    total_epochs: Optional[int]
    spec_sha256: str

    def __post_init__(self) -> None:
        if len(self.transition_steps) + 1 != len(self.sparsities):
            raise ValueError("Compiled schedule requires one fewer transition than sparsity values.")
        if any(left >= right for left, right in zip(self.transition_steps, self.transition_steps[1:])):
            raise ValueError("Compiled schedule transition steps must be strictly increasing.")
        if self.transition_steps and not 0 < self.transition_steps[-1] <= self.total_optimizer_steps:
            raise ValueError("Compiled schedule transitions must not exceed the training run.")

    def sparsity_at(self, completed_optimizer_steps: int) -> float:
        """Return sparsity for the next update after ``completed_optimizer_steps``."""

        if isinstance(completed_optimizer_steps, bool):
            raise ValueError("completed_optimizer_steps must be a non-negative integer.")
        completed_optimizer_steps = int(completed_optimizer_steps)
        if completed_optimizer_steps < 0:
            raise ValueError("completed_optimizer_steps must be non-negative.")
        return self.sparsities[bisect_right(self.transition_steps, completed_optimizer_steps)]

    def to_dict(self) -> dict:
        """Return JSON-safe runtime metadata for checkpoint manifests."""

        return {
            "schedule_spec": schedule_spec_to_dict(self.spec),
            "schedule_sha256": self.spec_sha256,
            "schedule_runtime": {
                "steps_per_epoch": self.steps_per_epoch,
                "planned_total_epochs": self.total_epochs,
                "planned_total_optimizer_steps": self.total_optimizer_steps,
                "transition_steps": list(self.transition_steps),
                "sparsities": list(self.sparsities),
            },
        }


def _stage_from_mapping(value: Any, *, index: int) -> BSAStage:
    if not isinstance(value, Mapping):
        raise ValueError(f"BSA schedule stage {index} must be a JSON object.")
    unknown = set(value) - {"name", "sparsity", "duration"}
    if unknown:
        raise ValueError(f"BSA schedule stage {index} has unknown fields: {sorted(unknown)}.")
    missing = {"sparsity", "duration"} - set(value)
    if missing:
        raise ValueError(f"BSA schedule stage {index} is missing fields: {sorted(missing)}.")
    return BSAStage(
        name=value.get("name"),
        sparsity=value["sparsity"],
        duration=value["duration"],
    )


def _spec_from_mapping(value: Any) -> BSAScheduleSpec:
    if not isinstance(value, Mapping):
        raise ValueError("BSA schedule JSON must contain an object at the top level.")
    unknown = set(value) - {"schema_version", "basis", "stages"}
    if unknown:
        raise ValueError(f"BSA schedule has unknown fields: {sorted(unknown)}.")
    missing = {"schema_version", "basis", "stages"} - set(value)
    if missing:
        raise ValueError(f"BSA schedule is missing fields: {sorted(missing)}.")
    stages = value["stages"]
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise ValueError("BSA schedule stages must be a JSON array.")
    return BSAScheduleSpec(
        schema_version=value["schema_version"],
        basis=value["basis"],
        stages=tuple(_stage_from_mapping(stage, index=index) for index, stage in enumerate(stages)),
    )


def _make_spec(*, basis: str, stages: Sequence[Mapping[str, Any]]) -> BSAScheduleSpec:
    return _spec_from_mapping(
        {"schema_version": _SUPPORTED_SCHEMA_VERSION, "basis": basis, "stages": stages}
    )


_PRESETS = MappingProxyType(
    {
        "legacy_progress_v1": _make_spec(
            basis="progress",
            stages=[
                {"sparsity": sparsity, "duration": "0.05"}
                for sparsity in ("0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7")
            ]
            + [{"name": "target", "sparsity": _TARGET, "duration": _REMAINDER}],
        ),
        "conservative_epoch_v1": _make_spec(
            basis="epoch",
            stages=[{"name": "dense-calibration", "sparsity": "0", "duration": "3"}]
            + [
                {"sparsity": sparsity, "duration": "1"}
                for sparsity in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.75")
            ]
            + [{"name": "target", "sparsity": _TARGET, "duration": _REMAINDER}],
        ),
        "mature_same_shape_v1": _make_spec(
            basis="epoch",
            stages=[{"name": "dense-calibration", "sparsity": "0", "duration": "1"}]
            + [
                {"sparsity": sparsity, "duration": "1"}
                for sparsity in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.75")
            ]
            + [{"name": "target", "sparsity": _TARGET, "duration": _REMAINDER}],
        ),
    }
)


def available_bsa_schedule_presets() -> Tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def load_bsa_schedule(value: _ScheduleInput) -> BSAScheduleSpec:
    """Load a preset name, ``@file.json``, inline JSON, mapping, or existing spec."""

    if isinstance(value, BSAScheduleSpec):
        return value
    if isinstance(value, Mapping):
        return _spec_from_mapping(value)
    if not isinstance(value, str):
        raise ValueError("BSA schedule must be a preset name, JSON object, @JSON path, or spec.")

    source = value.strip()
    if source in _PRESETS:
        return _PRESETS[source]
    if source.startswith("@"):
        path = Path(source[1:]).expanduser()
        if not path.is_file():
            raise ValueError(f"BSA schedule file does not exist: {path}")
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Unable to read BSA schedule file {path}: {error}") from error
    try:
        parsed = json.loads(source, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as error:
        presets = ", ".join(available_bsa_schedule_presets())
        raise ValueError(
            f"Unknown BSA schedule {value!r}. Use one of [{presets}], @file.json, or inline JSON."
        ) from error
    return _spec_from_mapping(parsed)


def schedule_spec_to_dict(value: _ScheduleInput) -> dict:
    """Return a deterministic JSON-safe representation with exact decimal strings."""

    spec = load_bsa_schedule(value)
    stages = []
    for stage in spec.stages:
        item = {
            "sparsity": (
                stage.sparsity
                if stage.sparsity == _TARGET
                else _decimal_text(stage.sparsity)
            ),
            "duration": (
                stage.duration
                if stage.duration == _REMAINDER
                else _decimal_text(stage.duration)
            ),
        }
        if stage.name is not None:
            item["name"] = stage.name
        stages.append(item)
    return {"schema_version": spec.schema_version, "basis": spec.basis, "stages": stages}


def canonical_schedule_spec(value: _ScheduleInput) -> str:
    """Return canonical JSON used to compare and hash schedule semantics."""

    return json.dumps(
        schedule_spec_to_dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def schedule_spec_sha256(value: _ScheduleInput) -> str:
    return hashlib.sha256(canonical_schedule_spec(value).encode("utf-8")).hexdigest()


def _resolve_run_size(
    *,
    steps_per_epoch: Optional[int],
    total_epochs: Optional[int],
    total_optimizer_steps: Optional[int],
) -> Tuple[Optional[int], Optional[int], int]:
    steps = None if steps_per_epoch is None else _positive_int(steps_per_epoch, field="steps_per_epoch")
    epochs = None if total_epochs is None else _positive_int(total_epochs, field="total_epochs")
    total = (
        None
        if total_optimizer_steps is None
        else _positive_int(total_optimizer_steps, field="total_optimizer_steps")
    )
    if total is None:
        if steps is None or epochs is None:
            raise ValueError(
                "Provide total_optimizer_steps, or both steps_per_epoch and total_epochs."
            )
        total = steps * epochs
    if steps is not None and epochs is not None and total != steps * epochs:
        raise ValueError(
            "total_optimizer_steps must equal steps_per_epoch * total_epochs when all are provided."
        )
    return steps, epochs, total


def _validate_duration_for_basis(duration: Decimal, *, basis: str) -> None:
    if basis in {"epoch", "optimizer_step"} and duration != duration.to_integral_value():
        raise ValueError(f"{basis} schedule durations must be whole numbers, received {duration}.")


def compile_bsa_schedule(
    value: _ScheduleInput,
    *,
    steps_per_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    total_optimizer_steps: Optional[int] = None,
    target_sparsity: Optional[Any] = None,
) -> CompiledBSASchedule:
    """Compile a schedule to exact completed-optimizer-step transitions.

    ``progress`` durations are multiplied using :class:`~decimal.Decimal` and
    rounded up, matching ``completed_steps / total_steps >= boundary`` without
    binary floating-point drift.
    """

    spec = load_bsa_schedule(value)
    steps, epochs, total = _resolve_run_size(
        steps_per_epoch=steps_per_epoch,
        total_epochs=total_epochs,
        total_optimizer_steps=total_optimizer_steps,
    )
    if spec.basis == "epoch" and (steps is None or epochs is None):
        raise ValueError("Epoch-based BSA schedules require steps_per_epoch and total_epochs.")

    target = (
        None
        if target_sparsity is None
        else _decimal(target_sparsity, field="target_sparsity")
    )
    if target is not None and not Decimal("0") <= target < Decimal("1"):
        raise ValueError(f"target_sparsity must be in [0, 1), received {target}.")
    if spec.stages[-1].sparsity == _TARGET:
        if target is None:
            raise ValueError(
                "This BSA schedule uses sparsity='target'; provide target_sparsity when compiling it."
            )
        spec = replace(
            spec,
            stages=spec.stages[:-1] + (replace(spec.stages[-1], sparsity=target),),
        )
    elif target is not None and spec.stages[-1].sparsity != target:
        raise ValueError(
            "BSA schedule final sparsity "
            f"{spec.stages[-1].sparsity} does not match target_sparsity {target}."
        )

    limit = {
        "epoch": Decimal(epochs) if epochs is not None else None,
        "progress": Decimal("1"),
        "optimizer_step": Decimal(total),
    }[spec.basis]
    assert limit is not None

    cumulative = Decimal("0")
    boundaries = []
    has_remainder = spec.stages[-1].duration == _REMAINDER
    numeric_stages = spec.stages[:-1] if has_remainder else spec.stages
    for stage in numeric_stages:
        assert isinstance(stage.duration, Decimal)
        _validate_duration_for_basis(stage.duration, basis=spec.basis)
        cumulative += stage.duration
        if cumulative >= limit and (has_remainder or stage is not spec.stages[-1]):
            raise ValueError(
                f"BSA schedule durations reach or exceed the {spec.basis} run length before the final stage."
            )
        if spec.basis == "epoch":
            boundary = int(cumulative) * steps  # type: ignore[operator]
        elif spec.basis == "optimizer_step":
            boundary = int(cumulative)
        else:
            boundary = int((cumulative * Decimal(total)).to_integral_value(rounding=ROUND_CEILING))
        boundaries.append(boundary)

    if has_remainder:
        if cumulative >= limit:
            raise ValueError("The final remainder stage must have a positive duration.")
    elif cumulative != limit:
        raise ValueError(
            f"BSA schedule durations sum to {cumulative}, but a {spec.basis} schedule must cover {limit}."
        )

    # A non-remainder schedule's final cumulative boundary is the end of the
    # run, not a transition to another stage.
    if not has_remainder:
        boundaries.pop()
    compiled_boundaries = []
    compiled_sparsities = [float(spec.stages[0].sparsity)]
    for boundary, next_stage in zip(boundaries, spec.stages[1:]):
        next_sparsity = float(next_stage.sparsity)
        if compiled_boundaries and boundary == compiled_boundaries[-1]:
            # Several progress thresholds can land on the same integer update
            # in tiny smoke runs.  At that update the old threshold function
            # skipped every zero-length stage and selected the last one.
            compiled_sparsities[-1] = next_sparsity
        else:
            compiled_boundaries.append(boundary)
            compiled_sparsities.append(next_sparsity)
    if any(left >= right for left, right in zip(compiled_boundaries, compiled_boundaries[1:])):
        raise ValueError("Compiled BSA schedule transition steps must be strictly increasing.")
    if compiled_boundaries and (
        compiled_boundaries[0] <= 0 or compiled_boundaries[-1] > total
    ):
        raise ValueError("Compiled BSA schedule transitions must not exceed the training run.")

    return CompiledBSASchedule(
        spec=spec,
        transition_steps=tuple(compiled_boundaries),
        sparsities=tuple(compiled_sparsities),
        total_optimizer_steps=total,
        steps_per_epoch=steps,
        total_epochs=epochs,
        spec_sha256=schedule_spec_sha256(spec),
    )


def compiled_bsa_schedule_from_dict(
    payload: Mapping[str, Any], *, target_sparsity: Optional[Any] = None
) -> CompiledBSASchedule:
    """Recompile and strictly validate schedule metadata from a checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("Compiled BSA schedule metadata must be a JSON object.")
    missing = {"schedule_spec", "schedule_sha256", "schedule_runtime"} - set(payload)
    if missing:
        raise ValueError(f"Compiled BSA schedule metadata is missing fields: {sorted(missing)}.")
    spec = load_bsa_schedule(payload["schedule_spec"])
    expected_hash = schedule_spec_sha256(spec)
    if payload["schedule_sha256"] != expected_hash:
        raise ValueError("BSA checkpoint schedule_sha256 does not match schedule_spec.")
    runtime = payload["schedule_runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("BSA checkpoint schedule_runtime must be a JSON object.")
    required_runtime = {
        "steps_per_epoch",
        "planned_total_epochs",
        "planned_total_optimizer_steps",
        "transition_steps",
        "sparsities",
    }
    missing_runtime = required_runtime - set(runtime)
    if missing_runtime:
        raise ValueError(
            f"BSA checkpoint schedule_runtime is missing fields: {sorted(missing_runtime)}."
        )
    compiled = compile_bsa_schedule(
        spec,
        steps_per_epoch=runtime["steps_per_epoch"],
        total_epochs=runtime["planned_total_epochs"],
        total_optimizer_steps=runtime["planned_total_optimizer_steps"],
        target_sparsity=target_sparsity,
    )
    if dict(runtime) != compiled.to_dict()["schedule_runtime"]:
        raise ValueError("BSA checkpoint schedule_runtime does not match the compiled schedule_spec.")
    return compiled


__all__ = [
    "BSAStage",
    "BSAScheduleSpec",
    "CompiledBSASchedule",
    "available_bsa_schedule_presets",
    "canonical_schedule_spec",
    "compile_bsa_schedule",
    "compiled_bsa_schedule_from_dict",
    "load_bsa_schedule",
    "schedule_spec_sha256",
    "schedule_spec_to_dict",
]
