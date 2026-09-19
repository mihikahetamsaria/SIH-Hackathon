import base64
import json
import cv2
import ollama


class QwenRecognizer:
    def __init__(self, model="qwen2.5vl:7b"):
        self.model = model

    def recognize(self, frame, procedure):
        steps = procedure.get("steps", [])
        reference = [
            {
                "step_id": step.get("step_id"),
                "name": step.get("name", ""),
                "instruction": step.get("instruction", ""),
            }
            for step in steps
        ]
        frame = cv2.resize(frame, (448, 252))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        if not ok:
            return self._empty_result("Could not encode camera frame")
        image = base64.b64encode(encoded.tobytes()).decode("utf-8")
        prompt = f'''You are a real-time visual action recognition system for a scientific experiment.
Identify the person's dominant physical action visible RIGHT NOW.
Use only visible evidence from hands, arms, body posture, and clearly visible objects.
Do not infer an action from the expected next step.
Reference procedure:
{json.dumps(reference, indent=2)}
Rules:
- Describe only what is visibly happening.
- Use 2 to 6 words for observed_action.
- If no meaningful action is visible, use "No visible action".
- If the image is unclear, use "Unclear".
- Do not guess hidden objects.
- Compare the visible action with every reference step.
- matched_step_id must be an exact reference step_id only when the visible action clearly matches that step.
- If the visible action is clear but does not match any reference step, set matched_step_id to null and recognized to true.
- If the image is unclear or there is no meaningful action, set matched_step_id to null and recognized to false.
- Never match based only on similar wording.
- "Hold phone" must not match "Hold pen".
- "Raise hand" must not automatically match "Give thumbs up".
- "Pointing" must not automatically match another hand gesture.
- Confidence represents visual certainty, not similarity to the procedure.
Return JSON only:
{{"observed_action":"<action>","matched_step_id":<step_id or null>,"confidence":<0.0-1.0>,"reason":"<brief visual evidence>","recognized":true or false}}'''
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image]}],
                options={"temperature": 0, "num_ctx": 2048, "num_predict": 80},
                keep_alive=-1,
            )
            text = response["message"]["content"].strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            result = json.loads(text)
            observed = str(result.get("observed_action") or "").strip()
            matched = result.get("matched_step_id")
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            unclear = observed.lower() in {"", "no visible action", "no visible movement", "unclear", "unknown", "loading screen", "hands not visible"}
            if unclear:
                return {"observed_action": observed, "matched_step_id": None, "confidence": min(confidence, 0.4), "reason": str(result.get("reason", "")).strip(), "recognized": False}
            if matched is not None:
                try:
                    matched = int(matched)
                except (TypeError, ValueError):
                    matched = None
            return {
                "observed_action": observed,
                "matched_step_id": matched,
                "confidence": confidence,
                "reason": str(result.get("reason", "")).strip(),
                "recognized": bool(result.get("recognized", True)),
            }
        except Exception as e:
            return self._empty_result(str(e))

    def _empty_result(self, reason):
        return {"observed_action": "", "matched_step_id": None, "confidence": 0.0, "reason": reason, "recognized": False}
