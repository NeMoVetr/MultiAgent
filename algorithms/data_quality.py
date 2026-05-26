import json
import math
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd


Number = float | int


class DataQuality:
    """
    Universal rule-based data quality processor.

    Example:
        dq = DataQuality(
            metadata_path="data_quality_metadata.json",
            state_path="data_quality_state.json",
        )

        clean_values = dq.clean(
            sensor_name="SONBEST_SM9560B",
            values={"illuminance_lux": 12000},
            context={"communication_ok": True, "sun_state": "night"},
        )

    By default, clean() returns a flat dictionary with numeric cleaned values.
    For diagnostics use return_details=True.
    """

    def __init__(
        self,
        metadata_path: str | Path = "data_quality_metadata.json",
        state_path: str | Path = "data_quality_state.json",
        autosave: bool = True,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent / "metadata"
        self.metadata_path = self._resolve_existing_path(metadata_path)
        self.state_path = self._resolve_state_path(state_path)
        self.autosave = autosave

        self.metadata = self._read_json(self.metadata_path)
        self.history_size = int(self.metadata.get("default_history_size", 5))
        self.state = self._load_or_create_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(
        self,
        sensor_name: str,
        values: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
    ) -> Dict[str, float] | Dict[str, Dict[str, Any]]:
        """
        Clean measurements for one sensor.

        Args:
            sensor_name: key from metadata["sensors"].
            values: raw measurements. Keys are parameter names from metadata.
            context: external context, for example:
                {
                    "communication_ok": True,
                    "sun_state": "day" | "twilight" | "night",
                    "polling_interval_minutes": 10,
                    "rain_or_irrigation": False,
                    "related_values": {"illuminance_lux": 50}
                }
            return_details: if True, returns cleaned values plus flags.

        Returns:
            By default: {parameter_name: numeric_clean_value}.
            With return_details=True:
                {
                    "values": {parameter_name: numeric_clean_value},
                    "flags": {parameter_name: [rule_notes]}
                }
        """
        context = context or {}
        sensor_cfg = self._get_sensor_config(sensor_name)
        self._ensure_sensor_state(sensor_name)

        communication_ok = bool(context.get("communication_ok", True))
        clean_values: Dict[str, float] = {}
        flags: Dict[str, List[str]] = {}
        counter_updates: Dict[str, float] = {}

        # If communication failed, we still return numeric values for all outputs.
        if not communication_ok:
            for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
                output_name = parameter_cfg.get("output_parameter", parameter_name)
                clean_value, note = self._missing_value_replacement(
                    sensor_name=sensor_name,
                    parameter_name=parameter_name,
                    output_name=output_name,
                    parameter_cfg=parameter_cfg,
                    context=context,
                )
                clean_values[output_name] = self._finalize_value(clean_value, parameter_cfg)
                flags.setdefault(output_name, []).append(note)

            clean_values, flags = self._apply_sensor_rules(sensor_name, sensor_cfg, clean_values, context, flags)
            self._update_histories(sensor_name, clean_values, counter_updates)
            if self.autosave:
                self.save_state()
            return {"values": clean_values, "flags": flags} if return_details else clean_values

        # Normal path: process every configured parameter by reusable rules.
        for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
            output_name = parameter_cfg.get("output_parameter", parameter_name)
            raw_value = values.get(parameter_name, values.get(output_name))
            value = self._to_number(raw_value)
            parameter_flags: List[str] = []

            if not self._is_number(value):
                value, note = self._missing_value_replacement(
                    sensor_name=sensor_name,
                    parameter_name=parameter_name,
                    output_name=output_name,
                    parameter_cfg=parameter_cfg,
                    context=context,
                )
                parameter_flags.append(f"missing_or_not_numeric -> {note}")
            else:
                value = float(value or 0.0)
                for rule in parameter_cfg.get("rules", []):
                    value, rule_note, counter_value = self._apply_parameter_rule(
                        sensor_name=sensor_name,
                        parameter_name=parameter_name,
                        output_name=output_name,
                        parameter_cfg=parameter_cfg,
                        rule=rule,
                        value=value,
                        all_clean_values=clean_values,
                        raw_values=values,
                        context=context,
                    )
                    if rule_note:
                        parameter_flags.append(rule_note)
                    if counter_value is not None:
                        counter_updates[parameter_name] = counter_value

            value = self._finalize_value(value, parameter_cfg)
            clean_values[output_name] = value
            flags[output_name] = parameter_flags or ["accepted"]

        clean_values, flags = self._apply_sensor_rules(sensor_name, sensor_cfg, clean_values, context, flags)
        self._update_histories(sensor_name, clean_values, counter_updates)

        if self.autosave:
            self.save_state()

        return {"values": clean_values, "flags": flags} if return_details else clean_values

    def clean_dataframe(
        self,
        sensor_name: str,
        frame: pd.DataFrame,
        context_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Optional helper for batch processing with pandas.

        Each row is passed through clean(). Context columns can include fields
        such as communication_ok, sun_state, polling_interval_minutes,
        rain_or_irrigation, probes_same_depth.
        """
        context_columns = context_columns or []
        cleaned_rows: List[Dict[str, float]] = []
        for _, row in frame.iterrows():
            context = {col: row[col] for col in context_columns if col in row.index}
            values = {col: row[col] for col in frame.columns if col not in context_columns}
            cleaned_rows.append(self.clean(sensor_name, values, context))  # type: ignore[arg-type]
        return pd.DataFrame(cleaned_rows, index=frame.index)

    def save_state(self) -> None:
        """Save current state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset_state_from_metadata(self) -> None:
        """Reset histories to initial values from metadata."""
        self.state = self._create_initial_state()
        if self.autosave:
            self.save_state()

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------
    def _apply_parameter_rule(
        self,
        sensor_name: str,
        parameter_name: str,
        output_name: str,
        parameter_cfg: Dict[str, Any],
        rule: Dict[str, Any],
        value: float,
        all_clean_values: Dict[str, float],
        raw_values: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, str, Optional[float]]:
        rule_type = rule.get("type")
        note = ""
        counter_update: Optional[float] = None

        if rule_type == "range_check":
            min_value = parameter_cfg.get("min")
            max_value = parameter_cfg.get("max")
            if min_value is not None and value < float(min_value):
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "range_check: below minimum -> replacement"
            elif max_value is not None and value > float(max_value):
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "range_check: above maximum -> replacement"

        elif rule_type == "negative_to_zero":
            if value < 0:
                value = 0.0
                note = "negative_to_zero: negative value -> 0"

        elif rule_type == "night_zero":
            threshold = float(rule.get("threshold", 0))
            if context.get("sun_state") == "night" and value > threshold:
                value = 0.0
                note = "night_zero: night value above threshold -> 0"

        elif rule_type == "jump_check":
            max_delta = float(rule["max_delta_from_median"])
            if self._should_apply_rule(rule, context):
                median_value = self._history_median(sensor_name, output_name)
                if abs(value - median_value) > max_delta:
                    value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                    note = "jump_check: deviation from history median -> replacement"

        elif rule_type == "rise_without_event_check":
            max_rise = float(rule["max_rise_from_median"])
            event_key = rule.get("event_context_key", "rain_or_irrigation")
            event_happened = bool(context.get(event_key, False))
            median_value = self._history_median(sensor_name, output_name)
            if not event_happened and (value - median_value) > max_rise:
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "rise_without_event_check: sharp rise without event -> replacement"

        elif rule_type == "fall_check":
            max_fall = float(rule["max_fall_from_median"])
            median_value = self._history_median(sensor_name, output_name)
            if (median_value - value) > max_fall:
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "fall_check: sharp fall -> replacement"

        elif rule_type == "related_parameter_minimum":
            related_parameter = rule["related_parameter"]
            related_value = all_clean_values.get(related_parameter)
            if related_value is None:
                related_value = self._to_number(raw_values.get(related_parameter))
            if not self._is_number(related_value):
                related_value = self._history_median(sensor_name, related_parameter)
            if float(related_value) < float(rule["minimum"]):
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "related_parameter_minimum: related value too low -> replacement"

        elif rule_type == "circular_jump_check":
            max_delta = float(rule["max_circular_delta_from_median"])
            median_value = self._history_median(sensor_name, output_name)
            circular_max = float(rule.get("circular_max", parameter_cfg.get("max", 360) + 1))
            delta = self._circular_distance(value, median_value, circular_max)
            if delta > max_delta:
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "circular_jump_check: circular deviation -> replacement"

        elif rule_type == "max_rate":
            interval_minutes = float(context.get("polling_interval_minutes", rule.get("default_interval_minutes", 1)))
            max_value = float(rule["max_per_minute"]) * interval_minutes
            if value > max_value:
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "max_rate: value exceeds allowed rate -> replacement"

        elif rule_type == "resolution_rounding":
            step = float(rule["step"])
            decimals = int(rule.get("decimals", self._decimals_from_step(step)))
            value = round(float(np.round(value / step) * step), decimals)
            note = f"resolution_rounding: rounded to step {step}"

        elif rule_type == "monotonic_counter_delta":
            previous_total = self._last_history_value(sensor_name, parameter_name)
            raw_total = value
            if raw_total >= previous_total:
                value = raw_total - previous_total
                note = "monotonic_counter_delta: interval = current total - previous total"
            else:
                value = 0.0
                note = "monotonic_counter_delta: counter reset or invalid decrease -> interval 0"
            counter_update = raw_total

        elif rule_type == "cross_sensor_check":
            if self._cross_sensor_condition_is_true(value, rule, context):
                value = self._replacement_value(sensor_name, output_name, parameter_cfg, rule, context)
                note = "cross_sensor_check: related measurement contradiction -> replacement"

        elif rule_type == "clamp":
            value = self._clamp(value, parameter_cfg)
            note = "clamp: final range limit applied"

        else:
            raise ValueError(f"Unknown rule type: {rule_type!r}")

        return float(value), note, counter_update

    def _apply_sensor_rules(
        self,
        sensor_name: str,
        sensor_cfg: Dict[str, Any],
        clean_values: Dict[str, float],
        context: Dict[str, Any],
        flags: Dict[str, List[str]],
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        for rule in sensor_cfg.get("sensor_rules", []):
            rule_type = rule.get("type")

            if rule_type == "multi_parameter_median":
                if not self._should_apply_rule(rule, context):
                    continue
                parameters = list(rule["parameters"])
                existing_values = [clean_values[p] for p in parameters if p in clean_values]
                if len(existing_values) < 3:
                    continue
                median_all = float(np.median(existing_values))
                max_delta = float(rule["max_delta_from_group_median"])
                for parameter in parameters:
                    if parameter not in clean_values:
                        continue
                    if abs(clean_values[parameter] - median_all) > max_delta:
                        other_values = [clean_values[p] for p in parameters if p != parameter and p in clean_values]
                        if other_values:
                            clean_values[parameter] = float(np.median(other_values))
                            flags.setdefault(parameter, []).append(
                                "multi_parameter_median: value replaced by median of other parameters"
                            )

            else:
                raise ValueError(f"Unknown sensor rule type: {rule_type!r}")

        # Always apply parameter bounds one final time.
        for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
            output_name = parameter_cfg.get("output_parameter", parameter_name)
            if output_name in clean_values:
                clean_values[output_name] = self._finalize_value(clean_values[output_name], parameter_cfg)

        return clean_values, flags

    # ------------------------------------------------------------------
    # Replacement and history helpers
    # ------------------------------------------------------------------
    def _replacement_value(
        self,
        sensor_name: str,
        output_name: str,
        parameter_cfg: Dict[str, Any],
        rule: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        replacement = rule.get("replacement", "median_history")
        if replacement == "zero":
            return 0.0
        if replacement == "median_history":
            return self._history_median(sensor_name, output_name)
        if replacement == "last_history":
            return self._last_history_value(sensor_name, output_name)
        if replacement == "min":
            return float(parameter_cfg.get("min", 0.0))
        if replacement == "max":
            return float(parameter_cfg.get("max", 0.0))
        if isinstance(replacement, (int, float)):
            return float(replacement)
        raise ValueError(f"Unknown replacement method: {replacement!r}")

    def _missing_value_replacement(
        self,
        sensor_name: str,
        parameter_name: str,
        output_name: str,
        parameter_cfg: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        # If a value must physically be zero at night, zero is safer than median.
        for rule in parameter_cfg.get("rules", []):
            if rule.get("type") == "night_zero" and context.get("sun_state") == "night":
                return 0.0, "communication/missing: night_zero -> 0"

        return self._history_median(sensor_name, output_name), "communication/missing: median of last 5 valid values"

    def _update_histories(
        self,
        sensor_name: str,
        clean_values: Dict[str, float],
        counter_updates: Dict[str, float],
    ) -> None:
        self._ensure_sensor_state(sensor_name)
        history = self.state["sensors"][sensor_name]["history"]

        for parameter_name, value in {**clean_values, **counter_updates}.items():
            if not self._is_number(value):
                continue
            values = history.setdefault(parameter_name, [])
            values.append(float(value))
            history[parameter_name] = values[-self.history_size :]

    def _history_median(self, sensor_name: str, parameter_name: str) -> float:
        history = self._get_history(sensor_name, parameter_name)
        return float(np.median(np.array(history, dtype=float)))

    def _last_history_value(self, sensor_name: str, parameter_name: str) -> float:
        history = self._get_history(sensor_name, parameter_name)
        return float(history[-1])

    def _get_history(self, sensor_name: str, parameter_name: str) -> List[float]:
        self._ensure_sensor_state(sensor_name)
        history = self.state["sensors"][sensor_name]["history"].get(parameter_name)
        if history:
            return [float(v) for v in history]

        # If somehow absent, restore from metadata or zeroes.
        parameter_cfg = self._find_parameter_cfg(sensor_name, parameter_name)
        initial = parameter_cfg.get("initial_history", [0] * self.history_size)
        history = [float(v) for v in initial][-self.history_size :]
        self.state["sensors"][sensor_name]["history"][parameter_name] = history
        return history

    # ------------------------------------------------------------------
    # Condition helpers
    # ------------------------------------------------------------------
    def _should_apply_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        required_context = rule.get("required_context", {})
        for key, expected_value in required_context.items():
            if context.get(key) != expected_value:
                return False
        return True

    def _cross_sensor_condition_is_true(self, value: float, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        required_context = rule.get("required_context", {})
        for key, expected_value in required_context.items():
            if context.get(key) != expected_value:
                return False

        if "value_above" in rule and not (value > float(rule["value_above"])):
            return False
        if "value_below" in rule and not (value < float(rule["value_below"])):
            return False

        related_values = context.get("related_values", {}) or {}
        for related_rule in rule.get("related", []):
            key = related_rule["key"]
            related_value = self._to_number(related_values.get(key, context.get(key)))
            if not self._is_number(related_value):
                return False
            related_value = float(related_value)
            if "below" in related_rule and not (related_value < float(related_rule["below"])):
                return False
            if "above" in related_rule and not (related_value > float(related_rule["above"])):
                return False
        return True

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------
    def _to_number(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            # pandas is intentionally used here because in real data pipelines
            # raw values often arrive as strings, numpy scalars or pandas values.
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        except Exception:
            return None
        if pd.isna(converted):
            return None
        return float(converted)

    def _is_number(self, value: Any) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _finalize_value(self, value: Any, parameter_cfg: Dict[str, Any]) -> float:
        if not self._is_number(value):
            # This branch should not be reached, but guarantees numeric output.
            value = float(parameter_cfg.get("min", 0.0))
        value = self._clamp(float(value), parameter_cfg)
        resolution = parameter_cfg.get("resolution")
        if resolution is not None:
            step = float(resolution)
            decimals = self._decimals_from_step(step)
            value = round(float(np.round(value / step) * step), decimals)
            value = self._clamp(value, parameter_cfg)
        return float(value)

    def _clamp(self, value: float, parameter_cfg: Dict[str, Any]) -> float:
        min_value = parameter_cfg.get("min")
        max_value = parameter_cfg.get("max")
        if min_value is not None or max_value is not None:
            low = -np.inf if min_value is None else float(min_value)
            high = np.inf if max_value is None else float(max_value)
            value = float(np.clip(value, low, high))
        return float(value)

    def _circular_distance(self, value: float, median_value: float, circular_max: float) -> float:
        direct = abs(value - median_value)
        wrapped = circular_max - direct
        return float(min(direct, wrapped))

    def _decimals_from_step(self, step: float) -> int:
        step_text = f"{step:.10f}".rstrip("0")
        if "." not in step_text:
            return 0
        return len(step_text.split(".")[1])

    # ------------------------------------------------------------------
    # Metadata / state helpers
    # ------------------------------------------------------------------
    def _resolve_existing_path(self, path: str | Path) -> Path:
        path = Path(path)
        candidates = [path, Path.cwd() / path, self.base_dir / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        # Let the file read raise a clear error.
        return (Path.cwd() / path).resolve()

    def _resolve_state_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        if (Path.cwd() / path).exists():
            return (Path.cwd() / path).resolve()
        if (self.base_dir / path).exists():
            return (self.base_dir / path).resolve()
        return (Path.cwd() / path).resolve()

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = self._read_json(self.state_path)
            self._fill_missing_state_from_metadata(state)
            return state
        return self._create_initial_state()

    def _create_initial_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"sensors": {}}
        for sensor_name, sensor_cfg in self.metadata.get("sensors", {}).items():
            state["sensors"][sensor_name] = {"history": {}}
            for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
                initial_history = parameter_cfg.get("initial_history", [0] * self.history_size)
                state["sensors"][sensor_name]["history"][parameter_name] = self._normalize_history(initial_history)

                output_name = parameter_cfg.get("output_parameter")
                if output_name:
                    output_initial = parameter_cfg.get("output_initial_history", initial_history)
                    state["sensors"][sensor_name]["history"][output_name] = self._normalize_history(output_initial)
        return state

    def _fill_missing_state_from_metadata(self, state: MutableMapping[str, Any]) -> None:
        state.setdefault("sensors", {})
        for sensor_name, sensor_cfg in self.metadata.get("sensors", {}).items():
            state["sensors"].setdefault(sensor_name, {"history": {}})
            state["sensors"][sensor_name].setdefault("history", {})
            history = state["sensors"][sensor_name]["history"]
            for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
                if parameter_name not in history:
                    history[parameter_name] = self._normalize_history(parameter_cfg.get("initial_history", [0] * self.history_size))
                output_name = parameter_cfg.get("output_parameter")
                if output_name and output_name not in history:
                    output_initial = parameter_cfg.get("output_initial_history", parameter_cfg.get("initial_history", [0] * self.history_size))
                    history[output_name] = self._normalize_history(output_initial)

    def _normalize_history(self, values: List[Any]) -> List[float]:
        normalized = [float(self._to_number(v) if self._to_number(v) is not None else 0.0) for v in values]
        if not normalized:
            normalized = [0.0] * self.history_size
        if len(normalized) < self.history_size:
            normalized = ([normalized[0]] * (self.history_size - len(normalized))) + normalized
        return normalized[-self.history_size :]

    def _ensure_sensor_state(self, sensor_name: str) -> None:
        if sensor_name not in self.state.get("sensors", {}):
            sensor_cfg = self._get_sensor_config(sensor_name)
            self.state.setdefault("sensors", {})[sensor_name] = {"history": {}}
            for parameter_name, parameter_cfg in sensor_cfg.get("parameters", {}).items():
                self.state["sensors"][sensor_name]["history"][parameter_name] = self._normalize_history(
                    parameter_cfg.get("initial_history", [0] * self.history_size)
                )
                output_name = parameter_cfg.get("output_parameter")
                if output_name:
                    output_initial = parameter_cfg.get("output_initial_history", parameter_cfg.get("initial_history", [0] * self.history_size))
                    self.state["sensors"][sensor_name]["history"][output_name] = self._normalize_history(output_initial)

    def _get_sensor_config(self, sensor_name: str) -> Dict[str, Any]:
        sensors = self.metadata.get("sensors", {})
        if sensor_name not in sensors:
            raise KeyError(f"Sensor {sensor_name!r} is not configured in metadata JSON")
        return sensors[sensor_name]

    def _find_parameter_cfg(self, sensor_name: str, parameter_name: str) -> Dict[str, Any]:
        sensor_cfg = self._get_sensor_config(sensor_name)
        parameters = sensor_cfg.get("parameters", {})
        if parameter_name in parameters:
            return parameters[parameter_name]
        for cfg in parameters.values():
            if cfg.get("output_parameter") == parameter_name:
                return cfg
        return {"initial_history": [0] * self.history_size, "min": None, "max": None}
