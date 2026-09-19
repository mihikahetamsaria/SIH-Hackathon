import base64
import json
import re
import time

import cv2
import numpy as np

try:
    import ollama
except ImportError:  # reported as a visible recognizer error instead of crashing the app
    ollama = None

IDLE_OBSERVATIONS = {"", "no visible action", "no visible movement", "unclear", "unknown",
                     "loading screen", "hands not visible"}


def _same_id(a, b):
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return False


class QwenRecognizer:
    """Zero-shot VLM recognizer (Ollama). Every failure is kept in `last_error`
    and returned in the result under "error", so the GUI can show it."""

    def __init__(self, model="qwen2.5vl:7b", verify=True, max_gap=1):
        self.model = model
        self.verify = verify          # second, strict yes/no look at the first frame of each new match
        self._max_gap = max_gap
        self.reset()
        self.last_raw = ""
        self.last_error = None
        self.last_latency_s = None
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ public
    def reset(self):
        """Forget the current match streak (call when a new experiment starts)."""
        self._streak_match = None
        self._streak_verdict = None   # None = not checked yet, True/False = strict check result
        self._streak_gap = 0
        self._streak_verify = None    # last verification details, kept for the whole streak

    def warmup(self):
        """Load the model into memory once, so the first real frame isn't a cold start.
        Returns {"ok": bool, "seconds": float, "error": str|None}."""
        t0 = time.time()
        if ollama is None:
            return {"ok": False, "seconds": 0.0, "error": "The 'ollama' package is not installed (pip install ollama)."}
        try:
            image = self._encode(np.zeros((252, 448, 3), np.uint8))
            ollama.chat(model=self.model,
                        messages=[{"role": "user", "content": "Reply with OK.", "images": [image]}],
                        options={"temperature": 0, "num_ctx": 2048, "num_predict": 1},
                        keep_alive=-1)
            return {"ok": True, "seconds": time.time() - t0, "error": None}
        except Exception as e:
            return {"ok": False, "seconds": time.time() - t0, "error": self._explain(e)}

    def recognize(self, frame, procedure):
        self.calls += 1
        steps = procedure.get("steps", [])
        reference = [
            {"step_id": s.get("step_id"), "name": s.get("name", ""), "instruction": s.get("instruction", "")}
            for s in steps
        ]
        if ollama is None:
            return self._fail("The 'ollama' package is not installed (pip install ollama).")
        try:
            image = self._encode(frame)
        except Exception as e:
            return self._fail(f"Could not encode camera frame: {e}")
        prompt = self._prompt(reference)
        t0 = time.time()
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image]}],
                options={"temperature": 0, "num_ctx": 2048, "num_predict": 120},
                keep_alive=-1,
            )
            text = response["message"]["content"].strip()
        except Exception as e:
            self.last_latency_s = time.time() - t0
            return self._fail(self._explain(e))
        self.last_latency_s = time.time() - t0
        self.last_raw = text
        result, salvaged = self._parse(text)
        if result is None:
            return self._fail(f"Model reply was not valid JSON: {text[:120]!r}")
        self.last_error = None

        observed = str(result.get("observed_action") or "").strip()
        matched = result.get("matched_step_id")
        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(result.get("reason", "")).strip()
        base = {"reason": reason, "error": None, "raw": text[:300], "latency_s": self.last_latency_s, "salvaged": salvaged}
        if observed.lower() in IDLE_OBSERVATIONS:
            self._note_no_match()
            return {**base, "observed_action": observed, "matched_step_id": None,
                    "confidence": min(confidence, 0.4), "recognized": False}
        if matched is not None:
            try:
                matched = int(matched)
            except (TypeError, ValueError):
                matched = None
        recognized = bool(result.get("recognized", True))
        extra = {"verified": None, "vetoed_step_id": None, "verify_reason": None}
        matched, observed, recognized, extra = self._verify_match(
            image, steps, matched, observed, recognized, extra, base)
        return {**base, **extra, "observed_action": observed, "matched_step_id": matched,
                "confidence": confidence, "recognized": recognized}

    # ------------------------------------------------------- strict verification
    def _note_no_match(self):
        self._streak_gap += 1
        if self._streak_gap > self._max_gap:
            self.reset()

    def _verify_match(self, image, steps, matched, observed, recognized, extra, base):
        """A 7B VLM often names the right step for the wrong reason (e.g. calls a phone a
        pen). So the first frame of every new match streak gets one strict yes/no check
        against that step. If it fails, the match is dropped and the frame is reported as
        a clear action that matches no step (-> UNEXPECTED in the controller)."""
        if matched is None:
            self._note_no_match()
            return matched, observed, recognized, extra
        self._streak_gap = 0
        if matched != self._streak_match:
            self._streak_match, self._streak_verdict, self._streak_verify = matched, None, None
        step = next((x for x in steps if _same_id(x.get("step_id"), matched)), None)
        if self.verify and step is not None and self._streak_verdict is None:
            verdict = self._strict_check(image, step)
            if verdict is not None:
                self._streak_verdict = verdict["matches"]
                self._streak_verify = verdict
                base["latency_s"] = (base.get("latency_s") or 0) + verdict["latency_s"]
        if self._streak_verdict is False:
            v = self._streak_verify or {}
            what = (v.get("visible_object") or "").strip()
            name = step.get("name", f"step {matched}") if step else f"step {matched}"
            extra.update({"verified": False, "vetoed_step_id": matched, "verify_reason": v.get("reason")})
            return None, (f"{what} - not '{name}'" if what and what.lower() != "none" else f"Not '{name}'"), True, extra
        extra.update({"verified": self._streak_verdict,
                      "verify_reason": (self._streak_verify or {}).get("reason")})
        return matched, observed, recognized, extra

    def _strict_check(self, image, step):
        prompt = (
            "You are checking one specific claim about the image.\n"
            f"Claim: the person is performing this experiment step: \"{step.get('name', '')}\" - {step.get('instruction', '')}\n"
            "Be strict. The claim is true only if the image clearly shows exactly this action, using every object "
            "named in the step (for example a pen is not a phone, and a red box is not a yellow box).\n"
            'Return JSON only: {"matches":true or false,"visible_object":"<main object held or used, or none>","reason":"<at most 8 words>"}'
        )
        t0 = time.time()
        try:
            response = ollama.chat(model=self.model,
                                   messages=[{"role": "user", "content": prompt, "images": [image]}],
                                   options={"temperature": 0, "num_ctx": 2048, "num_predict": 60},
                                   keep_alive=-1)
            text = response["message"]["content"].strip()
        except Exception as e:
            print("Verification error (match kept):", self._explain(e))
            return None
        m = re.search(r'"matches"\s*:\s*(true|false)', text)
        if not m:
            print("Verification reply not understood (match kept):", text[:100])
            return None
        vo = re.search(r'"visible_object"\s*:\s*"([^"]*)"', text)
        rs = re.search(r'"reason"\s*:\s*"([^"]*)', text)
        return {"matches": m.group(1) == "true", "visible_object": vo.group(1) if vo else "",
                "reason": rs.group(1) if rs else "", "latency_s": time.time() - t0, "raw": text[:200]}

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _encode(frame):
        frame = cv2.resize(frame, (448, 252))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        if not ok:
            raise ValueError("JPEG encode failed")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def _prompt(reference):
        # Field order matters: the fields the controller needs come FIRST, so a reply
        # that gets cut off at the token limit only loses the optional "reason".
        return f'''You are a real-time visual action recognition system for a scientific experiment.
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
- reason: at most 8 words.
Return JSON only, keys in exactly this order:
{{"observed_action":"<action>","matched_step_id":<step_id or null>,"confidence":<0.0-1.0>,"recognized":true or false,"reason":"<brief visual evidence>"}}'''

    @staticmethod
    def _parse(text):
        """Returns (dict|None, salvaged). Tolerates code fences, prose around the JSON,
        and replies truncated by the token limit."""
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        body = m.group(1).strip() if m else text
        start = body.find("{")
        if start >= 0:
            end = body.rfind("}")
            candidate = body[start:end + 1] if end > start else body[start:]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data, False
            except ValueError:
                pass
        out = {}
        m = re.search(r'"observed_action"\s*:\s*"([^"]*)"', body)
        if m:
            out["observed_action"] = m.group(1)
        m = re.search(r'"matched_step_id"\s*:\s*(null|"?-?\d+"?)', body)
        if m:
            out["matched_step_id"] = None if m.group(1) == "null" else m.group(1).strip('"')
        m = re.search(r'"confidence"\s*:\s*([0-9.]+)', body)
        if m:
            out["confidence"] = m.group(1)
        m = re.search(r'"recognized"\s*:\s*(true|false)', body)
        if m:
            out["recognized"] = m.group(1) == "true"
        m = re.search(r'"reason"\s*:\s*"([^"]*)', body)
        if m:
            out["reason"] = m.group(1)
        if "observed_action" in out:
            return out, True
        return None, False

    def _explain(self, e):
        msg = str(e)
        low = msg.lower()
        if isinstance(e, ConnectionError) or "connect" in low or "refused" in low:
            return "Cannot reach Ollama - start the Ollama app (or run `ollama serve`)."
        if getattr(e, "status_code", None) == 404 or "not found" in low:
            return f"Model '{self.model}' is not installed - run: ollama pull {self.model}"
        return f"{type(e).__name__}: {msg[:200]}"

    def _fail(self, message):
        self.failures += 1
        self.last_error = message
        print("Recognizer error:", message)
        return {"observed_action": "", "matched_step_id": None, "confidence": 0.0, "reason": message,
                "recognized": False, "error": message, "raw": self.last_raw[:300],
                "latency_s": self.last_latency_s, "salvaged": False}
