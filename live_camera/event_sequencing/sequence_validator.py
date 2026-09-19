import json
class ExperimentValidator:
    def __init__(self, experiment_file=None, reference_file=None):
        if reference_file:
            with open(reference_file, "r", encoding="utf-8") as f:
                reference = json.load(f)
            self.steps = [{"step_id": e["event_id"], "name": e["event"], "instruction": e["description"]} for e in reference["events"]]
        elif experiment_file:
            with open(experiment_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.steps = data.get("steps", [])
        else:
            raise ValueError("experiment_file or reference_file required")
        self.current_index = 0
        self.observed_events = []
    def process_event(self, step_id):
        if self.current_index >= len(self.steps):
            return {"status": "COMPLETE", "step_id": step_id, "current_expected_step_id": None}
        expected = self.steps[self.current_index]
        expected_id = expected["step_id"]
        completed_ids = set(self.observed_events)
        if step_id in completed_ids:
            return {"status": "IGNORED_REPEAT", "step_id": step_id, "current_expected_step_id": expected_id}
        if step_id == expected_id:
            completed = expected
            self.observed_events.append(step_id)
            self.current_index += 1
            if self.current_index >= len(self.steps):
                return {"status": "COMPLETE", "completed_step": completed, "current_expected_step_id": None}
            next_step = self.steps[self.current_index]
            return {"status": "CORRECT", "completed_step": completed, "current_expected_step_id": next_step["step_id"], "next_step": next_step}
        future_index = next((i for i, s in enumerate(self.steps[self.current_index + 1:], self.current_index + 1) if s["step_id"] == step_id), None)
        if future_index is not None:
            return {"status": "OUT_OF_ORDER", "expected": expected, "observed_step_id": step_id, "current_expected_step_id": expected_id, "corrective_action": expected["instruction"]}
        return {"status": "UNKNOWN", "expected": expected, "observed_step_id": step_id, "current_expected_step_id": expected_id, "corrective_action": expected["instruction"]}
    def build_gui_log(self):
        completed = set(self.observed_events)
        steps = []
        for step in self.steps:
            status = "done" if step["step_id"] in completed else "pending"
            steps.append({"step_id": step["step_id"], "name": step["name"], "status": status, "ts_start": None, "ts_end": None, "corrective_action": None})
        if self.current_index < len(self.steps):
            steps[self.current_index]["status"] = "in_progress"
        return {"current_expected_step_id": self.steps[self.current_index]["step_id"] if self.current_index < len(self.steps) else None, "steps": steps}