import threading
import time
import pyttsx3


class VoiceAlerts:
    def __init__(self):
        self._lock=threading.Lock()
        self._last_text=""
        self._last_time=0.0
        self._cooldown=3.0

    def speak(self,text):
        text=str(text).strip()

        if not text:
            return

        now=time.time()

        with self._lock:
            if text==self._last_text and now-self._last_time<self._cooldown:
                return

            self._last_text=text
            self._last_time=now

        threading.Thread(
            target=self._speak,
            args=(text,),
            daemon=True
        ).start()

    def _speak(self,text):
        try:
            engine=pyttsx3.init()
            engine.setProperty("rate",175)
            engine.setProperty("volume",1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("VOICE ERROR:",e)

    def shutdown(self):
        pass