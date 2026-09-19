"""
Offline voice alerts.

Why this is built the way it is
-------------------------------
The old version created a brand-new pyttsx3 engine in a brand-new thread for
every alert and swallowed every error. On Windows (SAPI5/COM) that pattern is
known to speak once and then go silent, and nothing told the operator.

Now:
  * ONE long-lived worker thread owns the engine (COM initialised once there).
  * Alerts go through a queue; if several pile up, only the newest is spoken.
  * If pyttsx3 fails, the OS speech command is used as a fallback
    (Windows System.Speech, macOS `say`, Linux espeak/spd-say) - all offline.
  * Failures are reported through `on_error` and `status()` instead of being
    hidden in the console.

Quick self-test (speaks three phrases, proves repeated alerts work):
    python -m GUI.voice_alerts
"""

import os
import platform
import queue
import shutil
import subprocess
import threading
import time


class VoiceAlerts:
    def __init__(self, rate=175, volume=1.0, cooldown=3.0, on_error=None):
        self.rate = rate
        self.volume = volume
        self._cooldown = cooldown
        self._on_error = on_error
        self._lock = threading.Lock()
        self._last_text = ""
        self._last_time = 0.0
        self._queue = queue.Queue()
        self._stop_speech = threading.Event()
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._engine = None
        self._next_engine_retry = 0.0
        self._error_reported = False
        self.backend = "starting"
        self.last_error = None
        self._thread = threading.Thread(target=self._worker, name="voice-alerts", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ public API
    def speak(self, text, force=False):
        """Queue speech. force=True bypasses duplicate suppression."""
        text = str(text).strip()
        if not text:
            return False
        now = time.time()
        with self._lock:
            if (not force and text == self._last_text and
                    now - self._last_time < self._cooldown):
                return False
            self._last_text = text
            self._last_time = now
        self._queue.put(text)
        return True

    def pause(self):
        self._paused.set()
        try:
            if self._engine is not None and hasattr(self._engine, "Pause"):
                self._engine.Pause()
        except Exception:
            pass
        return True

    def resume(self):
        self._paused.clear()
        try:
            if self._engine is not None and hasattr(self._engine, "Resume"):
                self._engine.Resume()
        except Exception:
            pass
        return True

    def is_paused(self):
        return self._paused.is_set()

    def stop(self):
        self._stop_speech.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(("__STOP__", True))
        except Exception:
            pass
        return True

    def status(self):
        return {"backend": self.backend, "last_error": self.last_error,
                "available": self.backend not in ("none", "starting")}

    def shutdown(self):
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=1.5)

    # ---------------------------------------------------------------- worker
    def _worker(self):
        self._engine=self._init_engine()
        if self._engine is not None:
            self.backend="sapi" if platform.system()=="Windows" else "pyttsx3"
        elif self._os_command_available():
            self.backend="os"
        else:
            self.backend="none"
        if self.backend == "none":
            self._report("No speech engine found (pyttsx3 failed and no OS speech command available).")
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            try:
                item = self._queue.get(timeout=0.25)
                if isinstance(item, tuple):
                    text, is_stop = item
                    if is_stop:
                        self._stop_current_speech()
                        self._stop_speech.clear()
                        continue
                else:
                    text = item
            except queue.Empty:
                continue
            if text is None:
                break
            if self._stop_speech.is_set():
                self._stop_speech.clear()
                self._stop_current_speech()
                continue
            if not self._say(text):
                self._report("Could not play voice alert.")

    def _init_engine(self):
        try:
            if platform.system() == "Windows":
                # COM must be initialised on the thread that owns the engine.
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    try:
                        import comtypes
                        comtypes.CoInitialize()
                    except Exception:
                        pass
            if platform.system() == "Windows":
                try:
                    import win32com.client
                    engine = win32com.client.Dispatch("SAPI.SpVoice")
                    engine.Rate = max(-10, min(10, int((self.rate - 175) / 15)))
                    engine.Volume = max(0, min(100, int(self.volume * 100)))
                    return engine
                except Exception as e:
                    self.last_error = f"SAPI init failed: {e}"
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            return engine
        except Exception as e:
            self.last_error = f"pyttsx3 init failed: {e}"
            print("VOICE ERROR:", self.last_error)
            return None

    def _say(self, text):
        if self._engine is not None:
            try:
                if hasattr(self._engine, "Speak"):
                    self._stop_speech.clear()
                    self._engine.Speak(text, 1)
                    while not self._stop_speech.is_set():
                        try:
                            if self._engine.WaitUntilDone(50):
                                break
                        except Exception:
                            time.sleep(0.05)
                    if self._stop_speech.is_set():
                        self._stop_current_speech()
                        self._stop_speech.clear()
                        return True
                else:
                    self._engine.say(text)
                    self._engine.runAndWait()
                return True
            except Exception as e:
                self.last_error = f"speech engine failed: {e}"
                print("VOICE ERROR:", self.last_error)
                try:
                    if hasattr(self._engine, "stop"):
                        self._engine.stop()
                except Exception:
                    pass
                self._engine = None
                self._next_engine_retry = 0.0
                self.backend = "os" if self._os_command_available() else "none"
        elif time.time() >= self._next_engine_retry:
            # engine was dropped after an error: retry (at most every 30 s)
            self._next_engine_retry = time.time() + 30.0
            self._engine = self._init_engine()
            if self._engine is not None:
                self.backend = "pyttsx3"
                return self._say(text)
        return self._say_with_os(text)

    def _stop_current_speech(self):
        try:
            if self._engine is not None and hasattr(self._engine, "Speak"):
                self._engine.Speak("", 3)
            elif self._engine is not None and hasattr(self._engine, "stop"):
                self._engine.stop()
        except Exception:
            pass

    # ------------------------------------------------------------ OS fallback
    @staticmethod
    def _os_command_available():
        system = platform.system()
        if system == "Windows":
            return shutil.which("powershell") is not None
        if system == "Darwin":
            return shutil.which("say") is not None
        return any(shutil.which(c) for c in ("espeak-ng", "espeak", "spd-say"))

    def _say_with_os(self, text):
        system=platform.system()
        try:
            if system=="Windows":
                cscript=shutil.which("cscript.exe") or shutil.which("cscript")
                if not cscript:
                    return False
                import tempfile
                script="""Dim voice
Set voice = CreateObject("SAPI.SpVoice")
voice.Rate = 0
voice.Volume = 100
voice.Speak WScript.Arguments(0), 0
"""
                with tempfile.NamedTemporaryFile("w",suffix=".vbs",delete=False,encoding="utf-8") as f:
                    f.write(script)
                    script_path=f.name
                try:
                    result=subprocess.run([cscript,"//nologo",script_path,text],timeout=30,capture_output=True,text=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                finally:
                    try:
                        os.remove(script_path)
                    except OSError:
                        pass
                if result.returncode!=0:
                    self.last_error=f"Windows SAPI failed: {result.stderr.strip() or result.stdout.strip()}"
                    print("VOICE ERROR:",self.last_error)
                    return False
                return True
            if system=="Darwin":
                result=subprocess.run(["say",text],timeout=20,capture_output=True)
                return result.returncode==0
            cmd=next((c for c in ("espeak-ng","espeak","spd-say") if shutil.which(c)),None)
            if cmd is None:
                return False
            result=subprocess.run([cmd,text],timeout=20,capture_output=True)
            return result.returncode==0
        except Exception as e:
            self.last_error=f"OS speech failed: {e}"
            print("VOICE ERROR:",self.last_error)
            return False

    # ------------------------------------------------------------ reporting
    def _report(self, message):
        detail = f"{message} {self.last_error or ''}".strip()
        self.last_error = detail
        if self._on_error and not self._error_reported:
            self._error_reported = True  # tell the operator once, not on every alert
            try:
                self._on_error(detail)
            except Exception:
                pass


if __name__ == "__main__":
    def _err(msg):
        print("PROBLEM:", msg)

    v = VoiceAlerts(on_error=_err)
    time.sleep(1.0)
    print("backend:", v.status())
    for phrase in ("Voice test one.", "Wrong step. Please perform: Give thumbs up.", "Voice test three."):
        print("speaking:", phrase)
        v.speak(phrase)
        time.sleep(4.0)
    print("final status:", v.status())
    v.shutdown()
