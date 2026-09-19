import json
import time
from pathlib import Path


class LiveController:
    """
    Central state controller for the live experiment.

    Qwen is responsible only for observation.
    LiveController decides whether that observation:
        - completes the current step
        - is out of order
        - is a wrong action
        - is uncertain
    """

    def __init__(self, experiment_file):
        self.experiment_file = Path(experiment_file)

        self.data = json.loads(
            self.experiment_file.read_text(encoding="utf-8")
        )

        self.steps = self.data.get("steps", [])

        self.current_index = 0

        # Last validated result
        self.last_result = None

        # Current experiment state
        self.current_state = {
            "status": "READY",
            "current_expected_step_id": self.current_expected_id(),
            "expected_action": self.current_expected_name(),
            "observed_action": None,
            "confidence": 0.0,
            "timestamp": None,
            "corrective_action": None,
        }

        # Session event history
        self.event_history = []

        self.log_path = (
            self.experiment_file.parent / "step_record_log.json"
        )

        self._write_state_log()

    # ---------------------------------------------------------
    # Main observation entry point
    # ---------------------------------------------------------

    def process_observation(
        self,
        observed_action,
        matched_step_id=None,
        confidence=0.0,
        recognized=True,
    ):
        """
        Process ONE observation from the recognition system.

        Qwen does not decide whether a step is complete.
        This controller makes that decision.
        """

        timestamp = time.time()

        # -----------------------------------------------------
        # Experiment already complete
        # -----------------------------------------------------

        if self.current_index >= len(self.steps):
            result = {
                "status": "COMPLETE",
                "observed_action": observed_action,
                "confidence": confidence,
                "timestamp": timestamp,
            }

            self.last_result = result
            return result

        expected = self.steps[self.current_index]

        expected_id = int(expected["step_id"])

        # Normalize matched step ID
        if matched_step_id is not None:
            try:
                matched_step_id = int(matched_step_id)
            except (TypeError, ValueError):
                matched_step_id = None

        confidence = max(
            0.0,
            min(1.0, float(confidence or 0.0))
        )

        # -----------------------------------------------------
        # UNCERTAIN OBSERVATION
        # -----------------------------------------------------

        if not recognized or matched_step_id is None:
            result = {
                "status": "UNCERTAIN",
                "expected": expected,
                "current_expected_step_id": expected_id,
                "expected_action": expected.get("name", ""),
                "observed_action": observed_action,
                "confidence": confidence,
                "timestamp": timestamp,
                "corrective_action": None,
            }

            self.current_state = result
            self.last_result = result

            # Uncertain observations are not experiment errors.
            # They should not advance the step.
            return result

        # -----------------------------------------------------
        # CORRECT CURRENT STEP
        # -----------------------------------------------------

        if matched_step_id == expected_id:
            return self.complete_step(
                expected_id,
                observed_action=observed_action,
                confidence=confidence,
                timestamp=timestamp,
            )

        # -----------------------------------------------------
        # OUT OF ORDER
        # -----------------------------------------------------

        future = next(
            (
                step
                for step in self.steps[self.current_index + 1:]
                if int(step["step_id"]) == matched_step_id
            ),
            None,
        )

        if future is not None:
            message = (
                f"Complete "
                f"'{expected.get('name', 'the expected step')}' "
                f"before "
                f"'{future.get('name', 'that step')}'."
            )

            return self._error(
                "OUT_OF_ORDER",
                message,
                observed_action,
                confidence,
                timestamp,
                matched_step_id,
            )

        # -----------------------------------------------------
        # RECOGNIZED BUT NOT PART OF EXPECTED PROCEDURE
        # -----------------------------------------------------

        message = (
            f"Please perform: "
            f"{expected.get('name', 'the expected step')}."
        )

        return self._error(
            "WRONG",
            message,
            observed_action,
            confidence,
            timestamp,
            matched_step_id,
        )

    # ---------------------------------------------------------
    # Step completion
    # ---------------------------------------------------------

    def complete_step(
        self,
        step_id,
        observed_action=None,
        confidence=0.0,
        timestamp=None,
    ):
        if self.current_index >= len(self.steps):
            return {
                "status": "COMPLETE",
                "observed_action": observed_action,
                "confidence": confidence,
                "timestamp": timestamp,
            }

        expected = self.steps[self.current_index]

        if int(step_id) != int(expected["step_id"]):
            return self._error(
                "WRONG",
                f"Please perform: "
                f"{expected.get('name', 'the expected step')}.",
                observed_action,
                confidence,
                timestamp or time.time(),
                step_id,
            )

        completed = expected

        # Advance only here.
        self.current_index += 1

        if self.current_index >= len(self.steps):
            status = "COMPLETE"
            next_expected_id = None
            next_action = None
        else:
            status = "CORRECT"
            next_expected_id = self.current_expected_id()
            next_action = self.current_expected_name()

        result = {
            "status": status,
            "completed_step": completed,
            "observed_action": observed_action,
            "confidence": confidence,
            "timestamp": timestamp or time.time(),
            "current_expected_step_id": next_expected_id,
            "expected_action": next_action,
            "corrective_action": None,
        }

        self.current_state = result
        self.last_result = result

        self._record_event(result)
        self._write_state_log()

        return result

    # ---------------------------------------------------------
    # Error
    # ---------------------------------------------------------

    def _error(
        self,
        status,
        message,
        observed_action=None,
        confidence=0.0,
        timestamp=None,
        observed_step_id=None,
    ):
        step = self.steps[self.current_index]

        result = {
            "status": status,
            "expected": step,
            "current_expected_step_id": int(step["step_id"]),
            "expected_action": step.get("name", ""),
            "observed_action": observed_action,
            "observed_step_id": observed_step_id,
            "confidence": confidence,
            "timestamp": timestamp or time.time(),
            "corrective_action": message,
        }

        self.current_state = result
        self.last_result = result

        self._record_event(result)
        self._write_state_log()

        return result

    # ---------------------------------------------------------
    # Current expected step
    # ---------------------------------------------------------

    def current_expected_id(self):
        if self.current_index >= len(self.steps):
            return None

        return int(self.steps[self.current_index]["step_id"])

    def current_expected_name(self):
        if self.current_index >= len(self.steps):
            return None

        return self.steps[self.current_index].get("name", "")

    # ---------------------------------------------------------
    # Event history
    # ---------------------------------------------------------

    def _record_event(self, result):
        event = {
            "timestamp": result.get("timestamp"),
            "status": result.get("status"),
            "expected_step_id": result.get(
                "current_expected_step_id"
            ),
            "expected_action": result.get(
                "expected_action"
            ),
            "observed_action": result.get(
                "observed_action"
            ),
            "observed_step_id": result.get(
                "observed_step_id"
            ),
            "confidence": result.get(
                "confidence", 0.0
            ),
            "correction": result.get(
                "corrective_action"
            ),
        }

        self.event_history.append(event)

    # ---------------------------------------------------------
    # State log
    # ---------------------------------------------------------

    def _write_state_log(self):
        now = time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        records = []

        for i, step in enumerate(self.steps):

            if i < self.current_index:
                status = "done"

            elif i == self.current_index:
                status = "in_progress"

            else:
                status = "pending"

            records.append(
                {
                    "step_id": step.get("step_id"),
                    "name": step.get("name", ""),
                    "status": status,
                    "ts_start": now
                    if i == self.current_index
                    else None,
                    "ts_end": now
                    if i < self.current_index
                    else None,
                }
            )

        data = {
            "current_expected_step_id":
                self.current_expected_id(),

            "current_expected_action":
                self.current_expected_name(),

            "steps": records,

            "events": self.event_history,
        }

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.log_path.write_text(
            json.dumps(
                data,
                indent=2
            ),
            encoding="utf-8",
        )

    # ---------------------------------------------------------
    # Public session information
    # ---------------------------------------------------------

    def get_state(self):
        return {
            "current_step_index":
                self.current_index,

            "total_steps":
                len(self.steps),

            "current_expected_step_id":
                self.current_expected_id(),

            "current_expected_action":
                self.current_expected_name(),

            "last_result":
                self.last_result,

            "event_history":
                list(self.event_history),
        }