import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import json

import cv2

from PySide6.QtCore import Qt,QTimer,QUrl,Signal
from PySide6.QtGui import QImage,QPixmap,QColor
from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QLabel,QPushButton,QScrollArea,QVBoxLayout,QHBoxLayout,QGraphicsDropShadowEffect,QFileDialog,QMessageBox

from GUI.log_reader import load_step_record_log,load_experiment_definition
from GUI.theme import THEME,build_stylesheet,hex_to_bgr,get_status_colors
from GUI import session_logger
from GUI.log_files_dialog import LogFilesDialog
from GUI.task_table_dialog import TaskTableDialog
from GUI.theme_dialog import ThemeDialog
from GUI.mistake_review_dialog import MistakeReviewDialog
from GUI.widgets_v3 import CircularProgress,StatusDot,StepRow
from GUI.voice_alerts import VoiceAlerts
from GUI.stream_sender import UDPFrameStreamer
from GUI.stream_dialog import StreamDialog
from GUI.experiment_summary_dialog import ExperimentSummaryDialog
from event_sequencing.live_controller import LiveController
from event_sequencing.qwen_recognizer import QwenRecognizer

DATA_DIR=Path(__file__).parent/"data"
STEP_LOG_PATH=DATA_DIR/"step_record_log.json"
EXPERIMENT_DEF_PATH=DATA_DIR/"experiment_definition.json"
REFERENCE_CLIPS_DIR=DATA_DIR/"reference_clips"
RECORDINGS_DIR=Path(__file__).parent/"recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
LOG_POLL_MS=100
VIDEO_POLL_MS=33
RECORDING_FPS=20.0
EVENT_BUFFER_SECONDS=3.0

class VideoOverlayContainer(QWidget):
    def __init__(self,video_label,bubble_label,parent=None):
        super().__init__(parent)
        self.video_label=video_label
        self.bubble_label=bubble_label
        self.experiment_finished=False
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(video_label)
        bubble_label.setParent(self)
        self._reposition()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._reposition()

    def _reposition(self):
        margin=18
        bubble=self.bubble_label
        bubble_w=min(360,self.width()-2*margin)
        bubble.setFixedWidth(bubble_w)
        bubble.adjustSize()
        bubble.move(self.width()-bubble.width()-margin,margin)

class MainWindowV3(QMainWindow):
    analysis_result=Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Guided Assistant - HAR On-board Experiment Validation")
        self.resize(1500,880)
        self.theme=dict(THEME)
        self.status_colors=get_status_colors(self.theme)
        self.experiment_def={}
        self.steps_def=[]
        self.step_lookup={}
        self._last_status_by_step={}
        self._step_rows={}
        self._status_dots={}
        self._overlay_status="pending"
        self._current_timer_text="--:--"
        self._current_step_name="-"
        self._last_observed_action=""
        self._last_expected_step_id=None
        self._last_analysis_time=0.0
        self.experiment_finished=False
        self._summary_shown=False
        self.current_experiment_path=None
        self.current_experiment_name=None
        self._summary_dialog=None
        self.event_buffer=deque()
        self.step_recordings={}
        self.event_recordings=[]
        self.session_started=datetime.now()
        self.session_archive_path=session_logger.new_session_path(self.session_started)
        self._theme_dialog=None
        self._mistake_dialog=None
        self._stream_dialog=None
        self.voice_alerts=VoiceAlerts()
        self.stream=UDPFrameStreamer()
        self._build_header()
        self._build_body()
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        if not self.capture.isOpened():
            self.video_label.setText("Camera failed to open")
        self.qwen_recognizer=QwenRecognizer()
        self.live_controller=None
        self._analysis_busy=False
        self._analysis_lock=threading.Lock()
        self.analysis_result.connect(self._handle_analysis_result)
        self.video_timer=QTimer(self)
        self.video_timer.timeout.connect(self._update_video_frame)
        self.video_timer.start(VIDEO_POLL_MS)
        self.log_timer=QTimer(self)
        self.log_timer.timeout.connect(self._poll_step_log)
        self.log_timer.start(LOG_POLL_MS)
        self._update_empty_state()

    def _add_shadow(self,widget,blur_radius=18,alpha=70,y_offset=5):
        shadow=QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QColor(0,0,0,alpha))
        shadow.setOffset(0,y_offset)
        widget.setGraphicsEffect(shadow)

    def _build_header(self):
        header=QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(56)
        layout=QHBoxLayout(header)
        layout.setContentsMargins(20,0,20,0)
        for label,handler in (
            ("PROJECTS",None),
            ("LOAD EXPERIMENT",self._load_experiment_json),
            ("LOGS",lambda:LogFilesDialog(self).exec()),
            ("ANALYTICS",lambda:TaskTableDialog(self).exec()),
            ("STREAM",self._open_stream_dialog),
            ("SETTINGS",self._open_theme_dialog),
        ):
            btn=QPushButton(label)
            btn.setObjectName("navLink")
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            if handler:
                btn.clicked.connect(handler)
            layout.addWidget(btn)
        layout.addStretch()
        self.header=header

    def _open_theme_dialog(self):
        dialog=ThemeDialog(self.theme,self._on_theme_changed,parent=self)
        dialog.show()
        self._theme_dialog=dialog

    def _open_stream_dialog(self):
        dialog=StreamDialog(self.stream,parent=self)
        dialog.exec()

    def _update_empty_state(self):
        self.experiment_title.setText("NO EXPERIMENT LOADED")
        self.current_action_label.setText("Current Action: -")
        self.validation_label.setText("Load an experiment JSON to begin.")
        self.completion_label.setText("Step Completion: -")
        self.task_progress_label.setText("Task Progress: - of -")
        self.progress_ring.set_value(0)
        self.repeat_btn.setEnabled(False)
        self._overlay_status="pending"
        self._current_timer_text="--:--"
        self._current_step_name="-"

    def _load_experiment_json(self):
        if self.live_controller is not None and not self.experiment_finished:
            answer=QMessageBox.question(self,"Load New Experiment","The current experiment is still running. Load another experiment and discard this run?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer!=QMessageBox.Yes:
                return
        if self._summary_dialog is not None:
            self._summary_dialog.close()
            self._summary_dialog=None
        path,_=QFileDialog.getOpenFileName(self,"Load Experiment JSON",str(DATA_DIR),"JSON Files (*.json)")
        if not path:
            return
        try:
            selected_path=Path(path)
            with selected_path.open("r",encoding="utf-8") as f:
                definition=json.load(f)
            steps=definition.get("steps")
            if not isinstance(steps,list) or not steps:
                raise ValueError("The JSON must contain a non-empty 'steps' list.")
            normalized=[]
            seen_ids=set()
            for i,step in enumerate(steps,1):
                if not isinstance(step,dict):
                    raise ValueError(f"Step {i} is not an object.")
                if "step_id" not in step or "name" not in step:
                    raise ValueError(f"Step {i} must contain 'step_id' and 'name'.")
                try:
                    step_id=int(step["step_id"])
                except (TypeError,ValueError):
                    raise ValueError(f"Step {i} has an invalid 'step_id'.")
                if step_id in seen_ids:
                    raise ValueError(f"Duplicate step_id: {step_id}.")
                seen_ids.add(step_id)
                item=dict(step)
                item["step_id"]=step_id
                normalized.append(item)
            definition["steps"]=normalized
            self._clear_to_initial_state()
            self.experiment_def=definition
            self.steps_def=normalized
            self.step_lookup={s["step_id"]:s for s in self.steps_def}
            self.current_experiment_path=selected_path
            self.current_experiment_name=definition.get("experiment_name",selected_path.stem)
            self._rebuild_step_panel()
            self._rebuild_dashboard_steps()
            self.experiment_title.setText(f"EXPERIMENT: {self.current_experiment_name}")
            self.validation_label.setText("Experiment selected. Upload reference videos or skip them to begin.")
            self._prompt_reference_videos()
        except Exception as e:
            self._clear_to_initial_state()
            QMessageBox.critical(self,"Invalid Experiment",f"Could not load experiment:\n\n{e}")

    def _prompt_reference_videos(self):
        box=QMessageBox(self)
        box.setWindowTitle("Reference Videos")
        box.setText("Reference videos are optional.")
        box.setInformativeText("For each step, you may upload a reference video. The required naming format is:\n\nstep_<step_id>.<extension>\n\nExample: step_1.mp4\n\nThe application will save each selected video using this format. Choose UPLOAD to add videos, SKIP ALL to start without them, or CANCEL to return to the initial screen.")
        upload=box.addButton("UPLOAD VIDEOS",QMessageBox.AcceptRole)
        skip=box.addButton("SKIP ALL",QMessageBox.DestructiveRole)
        cancel=box.addButton("CANCEL",QMessageBox.RejectRole)
        box.exec()
        clicked=box.clickedButton()
        if clicked is cancel:
            self._clear_to_initial_state()
            return
        if clicked is skip:
            self._start_loaded_experiment()
            return
        self._upload_reference_videos()
        self._start_loaded_experiment()

    def _start_loaded_experiment(self):
        if self.current_experiment_path is None or not self.steps_def:
            return
        try:
            self._reset_experiment_state()
            self.session_started=datetime.now()
            self.session_archive_path=session_logger.new_session_path(self.session_started)
            self.live_controller=LiveController(self.current_experiment_path)
            self.repeat_btn.setEnabled(True)
            self.reference_btn.setEnabled(True)
            self.abort_btn.setEnabled(True)
            self._poll_step_log()
        except Exception as e:
            self._clear_to_initial_state()
            QMessageBox.critical(self,"Experiment Error",f"Could not start experiment:\n\n{e}")

    def _clear_to_initial_state(self):
        if hasattr(self,"preview_player"):
            self.preview_player.stop()
        if hasattr(self,"preview_label"):
            self.preview_label.setText("Select a step to preview its reference clip")
        self.live_controller=None
        self.experiment_finished=False
        self.experiment_def={}
        self.steps_def=[]
        self.step_lookup={}
        self.current_experiment_path=None
        self.current_experiment_name=None
        self._reset_experiment_state()
        if hasattr(self,"step_panel_layout"):
            self._rebuild_step_panel()
        if hasattr(self,"dots_layout"):
            self._rebuild_dashboard_steps()
        if hasattr(self,"experiment_title"):
            self._update_empty_state()

    def _abort_experiment(self):
        if self.live_controller is None:
            return
        answer=QMessageBox.question(
            self,
            "Abort Experiment",
            "Are you sure you want to abort the current experiment?\n\nYour current run will be stopped and the application will return to the initial screen.",
            QMessageBox.Yes|QMessageBox.No,
            QMessageBox.No
        )
        if answer!=QMessageBox.Yes:
            return
        self._clear_to_initial_state()

    def _reset_experiment_state(self):
        self.experiment_finished=False
        self._summary_shown=False
        self._analysis_busy=False
        self._last_analysis_time=0.0
        self._last_observed_action=""
        self._last_expected_step_id=None
        self._overlay_status="pending"
        self._current_timer_text="--:--"
        self._current_step_name="-"
        self._last_status_by_step={}
        self.event_buffer.clear()
        self.step_recordings={}
        self.event_recordings=[]
        self.preview_player.stop()
        self.ai_bubble.hide()
        self.current_action_label.setText("Current Action: -")
        self.validation_label.setText("")
        self.completion_label.setText("Step Completion: 0%")
        self.task_progress_label.setText("Task Progress: - of -")
        self.progress_ring.set_value(0)
        self.repeat_btn.setEnabled(self.live_controller is not None)
        if hasattr(self,"reference_btn"):
            self.reference_btn.setEnabled(self.live_controller is not None)
        if hasattr(self,"abort_btn"):
            self.abort_btn.setEnabled(self.live_controller is not None)
        if hasattr(self,"abort_btn"):
            self.abort_btn.setEnabled(self.live_controller is not None)

    def _rebuild_step_panel(self):
        while self.step_panel_layout.count():
            item=self.step_panel_layout.takeAt(0)
            widget=item.widget()
            if widget:
                widget.deleteLater()
        self._step_rows={}
        for step in self.steps_def:
            row=StepRow(step,self.theme)
            self._step_rows[step["step_id"]]=row
            self.step_panel_layout.addWidget(row)
        self.step_panel_layout.addStretch()

    def _rebuild_dashboard_steps(self):
        while self.dots_layout.count():
            item=self.dots_layout.takeAt(0)
            widget=item.widget()
            if widget:
                widget.deleteLater()
        self._status_dots={}
        for step in self.steps_def:
            dot=StatusDot()
            self._status_dots[step["step_id"]]=dot
            self.dots_layout.addWidget(dot)

    def _reference_dir(self):
        if self.current_experiment_path is None:
            return REFERENCE_CLIPS_DIR
        path=REFERENCE_CLIPS_DIR/self.current_experiment_path.stem
        path.mkdir(parents=True,exist_ok=True)
        return path

    def _reference_clip_path(self,step_id):
        folder=self._reference_dir()
        matches=list(folder.glob(f"step_{step_id}.*"))
        return matches[0] if matches else folder/f"step_{step_id}.mp4"

    def _upload_reference_videos(self):
        if self.current_experiment_path is None or not self.steps_def:
            return
        folder=self._reference_dir()
        uploaded=0
        skipped=0
        for step in self.steps_def:
            step_id=int(step["step_id"])
            name=step.get("name",f"Step {step_id}")
            prompt=QMessageBox(self)
            prompt.setWindowTitle("Reference Video")
            prompt.setText(f"Reference video for Step {step_id}: {name}")
            prompt.setInformativeText(f"Naming format: step_{step_id}.mp4\n\nReference videos are optional. Upload one, skip this step, or cancel reference setup.")
            upload_btn=prompt.addButton("UPLOAD",QMessageBox.AcceptRole)
            skip_btn=prompt.addButton("SKIP",QMessageBox.DestructiveRole)
            cancel_btn=prompt.addButton("CANCEL SETUP",QMessageBox.RejectRole)
            prompt.exec()
            choice=prompt.clickedButton()
            if choice is cancel_btn:
                self._clear_to_initial_state()
                return
            if choice is skip_btn:
                skipped+=1
                continue
            existing=list(folder.glob(f"step_{step_id}.*"))
            if existing:
                answer=QMessageBox.question(
                    self,
                    "Reference Video Exists",
                    f"A reference video already exists for Step {step_id}: {name}.\n\nReplace it?",
                    QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel,
                    QMessageBox.No
                )
                if answer==QMessageBox.Cancel or answer==QMessageBox.No:
                    skipped+=1
                    continue
                for old in existing:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            path,_=QFileDialog.getOpenFileName(
                self,
                f"Reference video — Step {step_id}: {name}",
                str(folder),
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
            )
            if not path:
                skipped+=1
                continue
            source=Path(path)
            destination=folder/f"step_{step_id}{source.suffix.lower()}"
            try:
                destination.write_bytes(source.read_bytes())
                uploaded+=1
            except Exception as e:
                QMessageBox.critical(self,"Reference Video Error",f"Could not save Step {step_id}:\n\n{e}")
        if uploaded or skipped:
            self.preview_label.setText(f"Reference videos: {uploaded} uploaded, {skipped} skipped.")
            current_id=self._current_step_id()
            if current_id is not None:
                self._load_reference_clip(current_id)

    def _show_experiment_summary(self,steps,run_started_at=None,end_time=None):
        dialog=ExperimentSummaryDialog(
            experiment_name=self.current_experiment_name,
            steps=steps,
            event_recordings=self.event_recordings,
            start_time=run_started_at if run_started_at is not None else self.session_started,
            end_time=end_time,
            parent=self
        )
        dialog.view_log.connect(self._open_log_from_summary)
        dialog.run_again.connect(self._run_experiment_again)
        dialog.load_new.connect(self._load_experiment_json)
        dialog.show()
        self._summary_dialog=dialog

    def _open_log_from_summary(self):
        LogFilesDialog(self).exec()

    def _run_experiment_again(self):
        if self.current_experiment_path is None:
            return
        if self._summary_dialog:
            self._summary_dialog.close()
            self._summary_dialog=None
        self._reset_experiment_state()
        self.session_started=datetime.now()
        self.session_archive_path=session_logger.new_session_path(self.session_started)
        self.live_controller=LiveController(self.current_experiment_path)
        self.abort_btn.setEnabled(True)
        self._poll_step_log()

    def _on_theme_changed(self,theme):
        self.theme=theme
        self.status_colors=get_status_colors(self.theme)
        QApplication.instance().setStyleSheet(build_stylesheet(self.theme)+self._extra_stylesheet())
        self.video_label.setStyleSheet("background-color:#000;border-radius:10px;")
        self._refresh_all_step_rows()

    def _build_body(self):
        central=QWidget()
        root=QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
        root.addWidget(self.header)
        body_row=QHBoxLayout()
        body_row.setContentsMargins(16,16,16,8)
        body_row.setSpacing(16)
        body_row.addWidget(self._build_step_panel(),stretch=3)
        body_row.addWidget(self._build_video_panel(),stretch=5)
        body_row.addWidget(self._build_preview_panel(),stretch=2)
        root.addLayout(body_row,stretch=1)
        root.addWidget(self._build_dashboard())
        self.setCentralWidget(central)
        QApplication.instance().setStyleSheet(build_stylesheet(self.theme)+self._extra_stylesheet())

    def _extra_stylesheet(self):
        t=self.theme
        return f"""
        #headerBar {{ background-color:{t['panel_bg']}; border-bottom:1px solid {t['border']}; }}
        QPushButton#navLink {{ background:transparent; border:none; font-weight:600; padding:6px 10px; color:{t['text']}; }}
        QPushButton#navLink:hover {{ color:{t['accent']}; }}
        #panelCard {{ background-color:{t['panel_bg']}; border-radius:12px; }}
        #dashboardBar {{ background-color:{t['panel_bg']}; border-top:1px solid {t['border']}; }}
        """

    def _build_step_panel(self):
        card=QWidget()
        card.setObjectName("panelCard")
        outer=QVBoxLayout(card)
        outer.setContentsMargins(14,14,14,14)
        self.experiment_title=QLabel(f"EXPERIMENT: {self.current_experiment_name}")
        self.experiment_title.setObjectName("sectionTitle")
        outer.addWidget(self.experiment_title)
        title_row=QHBoxLayout()
        title=QLabel("GUIDED TASK PROCEDURE")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        self.reference_btn=QPushButton("UPLOAD REFERENCE VIDEOS")
        self.reference_btn.clicked.connect(self._upload_reference_videos)
        self.reference_btn.setEnabled(False)
        title_row.addWidget(self.reference_btn)
        outer.addLayout(title_row)
        scroll=QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner=QWidget()
        inner_layout=QVBoxLayout(inner)
        self.step_panel_layout=inner_layout
        inner_layout.setSpacing(10)
        for step in self.steps_def:
            row=StepRow(step,self.theme)
            self._step_rows[step["step_id"]]=row
            inner_layout.addWidget(row)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll,stretch=1)
        self._add_shadow(card)
        return card

    def _build_video_panel(self):
        card=QWidget()
        card.setObjectName("panelCard")
        layout=QVBoxLayout(card)
        layout.setContentsMargins(10,10,10,10)
        self.video_label=QLabel("No camera feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600,480)
        self.video_label.setStyleSheet("background-color:#000;border-radius:10px;")
        self.ai_bubble=QLabel("")
        self.ai_bubble.setObjectName("bubbleLabel")
        self.ai_bubble.setWordWrap(True)
        self.ai_bubble.hide()
        container=VideoOverlayContainer(self.video_label,self.ai_bubble)
        layout.addWidget(container)
        self._add_shadow(card)
        self.video_container=container
        return card

    def _build_preview_panel(self):
        card=QWidget()
        card.setObjectName("panelCard")
        layout=QVBoxLayout(card)
        layout.setContentsMargins(14,14,14,14)
        title=QLabel("PREVIEW AREA")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.preview_video=QVideoWidget()
        self.preview_video.setMinimumHeight(200)
        layout.addWidget(self.preview_video,stretch=1)
        self.preview_player=QMediaPlayer(self)
        self.preview_audio=QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_video)
        self.preview_label=QLabel("Select a step to preview its reference clip")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        controls=QHBoxLayout()
        play_btn=QPushButton("PLAY")
        pause_btn=QPushButton("PAUSE")
        play_btn.clicked.connect(self.preview_player.play)
        pause_btn.clicked.connect(self.preview_player.pause)
        controls.addWidget(play_btn)
        controls.addWidget(pause_btn)
        layout.addLayout(controls)
        demo_btn=QPushButton("Watch a Quick Demo")
        demo_btn.setObjectName("demoBtn")
        demo_btn.clicked.connect(self._play_current_reference_clip)
        layout.addWidget(demo_btn)
        self._add_shadow(card)
        return card

    def _build_dashboard(self):
        bar=QWidget()
        bar.setObjectName("dashboardBar")
        layout=QHBoxLayout(bar)
        layout.setContentsMargins(24,14,24,14)
        layout.setSpacing(24)
        left=QVBoxLayout()
        self.current_action_label=QLabel("Current Action: -")
        self.current_action_label.setStyleSheet("font-size:20px;font-weight:800;")
        left.addWidget(self.current_action_label)
        self.validation_label=QLabel("")
        self.validation_label.setWordWrap(True)
        left.addWidget(self.validation_label)
        layout.addLayout(left,stretch=3)
        mid=QVBoxLayout()
        self.completion_label=QLabel("Step Completion: 0%")
        self.completion_label.setStyleSheet("font-weight:700;")
        mid.addWidget(self.completion_label)
        dots_row=QHBoxLayout()
        self.dots_layout=dots_row
        for step in self.steps_def:
            dot=StatusDot()
            self._status_dots[step["step_id"]]=dot
            dots_row.addWidget(dot)
        mid.addLayout(dots_row)
        self.task_progress_label=QLabel("Task Progress: - of -")
        self.task_progress_label.setStyleSheet("font-size:12px;")
        mid.addWidget(self.task_progress_label)
        layout.addLayout(mid,stretch=3)
        btn_col=QVBoxLayout()
        self.repeat_btn=QPushButton("REPEAT STEP")
        self.repeat_btn.clicked.connect(self._on_repeat_step)
        btn_col.addWidget(self.repeat_btn)
        self.abort_btn=QPushButton("ABORT EXPERIMENT")
        self.abort_btn.clicked.connect(self._abort_experiment)
        self.abort_btn.setEnabled(False)
        btn_col.addWidget(self.abort_btn)
        btn_col.addStretch()
        layout.addLayout(btn_col,stretch=2)
        self.progress_ring=CircularProgress()
        self.progress_ring.set_colors(self.theme["border"],self.theme["success"],self.theme["text"])
        layout.addWidget(self.progress_ring,stretch=1)
        self._add_shadow(bar,blur_radius=14,alpha=50,y_offset=-2)
        return bar

    def _update_video_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok,frame=self.capture.read()
        if not ok:
            return
        now=time.time()
        self.event_buffer.append((now,frame.copy()))
        cutoff=now-EVENT_BUFFER_SECONDS
        while self.event_buffer and self.event_buffer[0][0]<cutoff:
            self.event_buffer.popleft()
        if self.live_controller is not None and not self.experiment_finished:
            with self._analysis_lock:
                if not self._analysis_busy and now-self._last_analysis_time>=0.5:
                    self._analysis_busy=True
                    self._last_analysis_time=now
                    threading.Thread(target=self._run_live_analysis,args=(frame.copy(),),daemon=True).start()
        self._draw_overlay(frame)
        self.stream.maybe_send(frame,{"step":self._current_step_name,"status":self._overlay_status,"elapsed":self._current_timer_text,"ts":datetime.now().isoformat(timespec="seconds")})
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb.shape
        qimg=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.size(),Qt.KeepAspectRatio,Qt.FastTransformation))

    def _run_live_analysis(self,frame):
        try:
            if self.experiment_finished or self.live_controller is None:
                return
            result=self.qwen_recognizer.recognize(frame,self.live_controller.data)
            self.analysis_result.emit(result)
        except Exception as e:
            print("Qwen error:",e)
        finally:
            with self._analysis_lock:
                self._analysis_busy=False

    def _handle_analysis_result(self,result):
        if self.live_controller is None or self.experiment_finished:
            return
        observed=str(result.get("observed_action") or "").strip()
        expected_id=self._current_step_id()
        if expected_id!=self._last_expected_step_id:
            self._last_expected_step_id=expected_id
            self._last_observed_action=""
        if observed:
            self._last_observed_action=observed
            self.current_action_label.setText(f"Current Action: {observed.title()}")
        validation=self.live_controller.process_observation(observed_action=observed,matched_step_id=result.get("matched_step_id"),confidence=result.get("confidence",0.0),recognized=result.get("recognized",False))
        print(f"Validation: {validation}")
        if validation.get("status")=="UNCERTAIN":
            self._overlay_status="pending"
            self.validation_label.setText("Validation: Waiting for a clear action...")
            return
        if validation.get("status") in ("CORRECT","COMPLETE"):
            completed=validation.get("completed_step")
            if completed:
                self._save_event_clip(completed["step_id"],"CORRECT")
            self._overlay_status="done"
            completed=validation.get("completed_step",{})
            self._show_ai_bubble(f"Nicely done: {completed.get('name','Step complete')}.","done")
            self.validation_label.setText(f"Validation: Step complete. Observed: {observed}.")
            QApplication.beep()
        elif validation.get("status") in ("WRONG","OUT_OF_ORDER","UNEXPECTED"):
            self._save_event_clip(validation["expected"]["step_id"],validation["status"])
            self._handle_dynamic_error(validation)
        elif validation.get("status")=="IGNORED_REPEAT":
            self._overlay_status="done"
        elif validation.get("status")=="PENDING_CONFIRMATION":
            self._overlay_status="pending"
            self.validation_label.setText(f"Validation: confirming action ({validation.get('confirmations',0)}/{validation.get('required_confirmations',3)})...")

    def _handle_dynamic_error(self,validation):
        status=validation.get("status","WRONG")
        expected=validation.get("expected",{})
        observed=validation.get("observed_action","unknown action")
        expected_name=expected.get("name","the expected step")
        self._overlay_status="wrong"
        if status=="OUT_OF_ORDER":
            spoken=f"Wrong step. Please perform: {expected_name}."
        elif status=="UNEXPECTED":
            spoken=f"Unexpected action. Please perform: {expected_name}."
        else:
            spoken=f"Wrong action. Please perform: {expected_name}."
        self._show_ai_bubble(spoken,"wrong")
        self.validation_label.setText(f"Validation: {spoken}")
        self.voice_alerts.speak(spoken)
        step_id=expected.get("step_id")
        if step_id is not None:
            self._auto_show_reference(step_id)

    def _draw_overlay(self,frame):
        color=hex_to_bgr(self.status_colors.get(self._overlay_status,self.theme["pending"]))
        h,w=frame.shape[:2]
        cv2.rectangle(frame,(0,0),(w-1,h-1),color,6)
        self._draw_timer_badge(frame,color)

    def _draw_timer_badge(self,frame,color):
        text=self._current_timer_text
        h,w=frame.shape[:2]
        font=cv2.FONT_HERSHEY_SIMPLEX
        scale,thickness=0.9,2
        (tw,th),_=cv2.getTextSize(text,font,scale,thickness)
        pad=12
        x2,y2=w-16,h-16
        x1,y1=x2-tw-2*pad,y2-th-2*pad
        overlay=frame.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,0,0),-1)
        cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,text,(x1+pad,y2-pad),font,scale,(255,255,255),thickness,cv2.LINE_AA)

    def _poll_step_log(self):
        if self.live_controller is None:
            return
        log_path=self.live_controller.log_path
        data=load_step_record_log(log_path)
        if data is None:
            return
        steps=data.get("steps",[])
        current_id=data.get("current_expected_step_id")
        current_step=self.step_lookup.get(current_id,{})
        total=len(steps) if steps else 1
        done_count=sum(1 for s in steps if s["status"]=="done")
        if steps and done_count==len(steps):
            if not self.experiment_finished:
                self.experiment_finished=True
                self._analysis_busy=False
                self._overlay_status="done"
                self._current_timer_text="--:--"
                self.current_action_label.setText("Current Action: Experiment Complete")
                self.validation_label.setText("Validation: Experiment completed successfully.")
                self._show_ai_bubble("Experiment complete.","done")
                self._last_observed_action=""
                if not self._summary_shown:
                    self._summary_shown=True
                    QTimer.singleShot(500,lambda:self._show_experiment_summary(steps))
        step_index=next((i for i,s in enumerate(steps) if s["step_id"]==current_id),len(steps)-1)
        self._current_step_name=current_step.get("name","-")
        pct=int(done_count/total*100)
        self.completion_label.setText(f"Step Completion: {pct}%")
        self.task_progress_label.setText(f"Task Progress: Step {step_index+1} of {total}")
        self.progress_ring.set_value(pct)
        active_record=next((s for s in steps if s["status"]=="in_progress"),None)
        if active_record is None:
            active_record=next((s for s in steps if s["step_id"]==current_id),None)
        self._current_timer_text=self._timer_text_for(active_record) if active_record else "--:--"
        self._update_step_rows(steps,current_id)
        self._update_status_dots(steps,current_id)
        self._handle_status_transitions(steps)
        self._persist_archive(steps)

    def _update_step_rows(self,steps,current_id):
        for s in steps:
            row=self._step_rows.get(s["step_id"])
            if not row:
                continue
            is_current=s["step_id"]==current_id or s["status"]=="in_progress"
            timer_text=self._timer_text_for(s)
            row.set_state(s["status"],is_current,timer_text,self.status_colors)
            if is_current:
                hint=self.step_lookup.get(s["step_id"],{}).get("corrective_message","Follow the on-screen guidance for this step.")
                row.set_hint(hint)

    def _timer_text_for(self,step_record):
        ts_start=step_record.get("ts_start")
        ts_end=step_record.get("ts_end")
        if step_record["status"]=="in_progress" and ts_start:
            elapsed=int(time.time()-self._parse_ts(ts_start))
            return f"0:{elapsed:02d}" if elapsed<60 else f"{elapsed//60}:{elapsed%60:02d}"
        if ts_start and ts_end:
            dur=int(self._parse_ts(ts_end)-self._parse_ts(ts_start))
            return f"0:{dur:02d}" if dur<60 else f"{dur//60}:{dur%60:02d}"
        return "--:--"

    @staticmethod
    def _parse_ts(ts):
        try:
            if isinstance(ts,(int,float)):
                return float(ts)
            return datetime.fromisoformat(str(ts)).timestamp()
        except (ValueError,TypeError,OverflowError):
            return time.time()

    def _update_status_dots(self,steps,current_id):
        for s in steps:
            dot=self._status_dots.get(s["step_id"])
            if not dot:
                continue
            color=self.status_colors.get(s["status"],self.theme["pending"])
            dot.set_status(s["status"],color,self.theme["border"])

    def _refresh_all_step_rows(self):
        if self.live_controller is None:
            return
        data=load_step_record_log(self.live_controller.log_path)
        if data:
            self._update_step_rows(data.get("steps",[]),data.get("current_expected_step_id"))
            self._update_status_dots(data.get("steps",[]),data.get("current_expected_step_id"))

    def _persist_archive(self,steps):
        enriched=[]
        for s in steps:
            video_path=self.step_recordings.get(s["step_id"])
            enriched.append({**s,"video_path":str(video_path) if video_path else None})
        session_logger.save_session(
            self.session_archive_path,
            self.session_started,
            enriched,
            self.event_recordings
        )

    def _handle_status_transitions(self,steps):
        for s in steps:
            step_id=s["step_id"]
            status=s["status"]
            prev=self._last_status_by_step.get(step_id)
            if status!=prev:
                self._on_status_changed(s,prev)
            self._last_status_by_step[step_id]=status

    def _on_status_changed(self,step,prev_status):
        step_id=step["step_id"]
        status=step["status"]
        if status=="in_progress":
            self._overlay_status="in_progress"
            self._show_ai_bubble(f"I see the {step.get('name','step')}. Go ahead.",status)
            self.validation_label.setText(f"Validation: In progress. {step['name']}.")
        elif status=="done":
            self._overlay_status="done"
            self._show_ai_bubble(f"Nicely done: {step['name']}.",status)
            self.validation_label.setText(f"Validation: Step complete. {step['name']}.")
        elif status in ("skipped","out_of_order","wrong"):
            message=step.get("corrective_action") or self.step_lookup.get(step_id,{}).get("corrective_message",f"Please perform: {step.get('name','the expected step')}.")
            self._overlay_status=status
            self._show_ai_bubble(message,status)
            self.validation_label.setText(f"Validation: {status.replace('_',' ').title()}. Expected: {step.get('name','')}. {message}")
            self._trigger_voice_alert(step,status,message)
            self._auto_show_reference(step_id)

    def _trigger_voice_alert(self,step,status,message):
        step_name=step.get("name",f"step {step.get('step_id','')}")
        if status=="out_of_order":
            spoken=f"Wrong step. Please perform: {step_name}."
        else:
            spoken=f"Wrong action. Please perform: {step_name}."
        QApplication.beep()
        QTimer.singleShot(180,QApplication.beep)
        QTimer.singleShot(500,lambda:self.voice_alerts.speak(spoken))

    def _open_mistake_review(self,step,message):
        step_id=step["step_id"]
        your_clip=self.step_recordings.get(step_id)
        reference_clip=self._reference_clip_path(step_id)
        dialog=MistakeReviewDialog(
            step_name=step.get("name",f"Step {step_id}"),
            message=message,
            your_clip_path=your_clip,
            reference_clip_path=reference_clip if reference_clip.exists() else None,
            parent=self
        )
        dialog.show()
        self._mistake_dialog=dialog

    def _show_ai_bubble(self,text,status):
        color=self.status_colors.get(status,self.theme["accent"])
        self.ai_bubble.setText(text)
        self.ai_bubble.setStyleSheet(f"background-color:rgba(20,20,20,220);color:white;border:1px solid {color};border-radius:12px;padding:12px;font-size:13px;")
        self.ai_bubble.show()
        self.ai_bubble.adjustSize()
        QTimer.singleShot(4000,self.ai_bubble.hide)

    def _save_event_clip(self,step_id,status):
        if not self.event_buffer:
            return
        frames=list(self.event_buffer)
        w=frames[0][1].shape[1]
        h=frames[0][1].shape[0]
        path=RECORDINGS_DIR/f"step_{step_id}_{status.lower()}_{int(time.time())}.mp4"
        writer=cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            RECORDING_FPS,
            (w,h)
        )
        for _,frame in frames:
            writer.write(frame)
        writer.release()
        self.event_recordings.append({
            "step_id":step_id,
            "status":status,
            "video_path":str(path)
        })
        if status=="CORRECT":
            self.step_recordings[step_id]=path

    def _current_step_id(self):
        if self.live_controller is None:
            return None
        data=load_step_record_log(self.live_controller.log_path)
        return data.get("current_expected_step_id") if data else None

    def _auto_show_reference(self,step_id):
        self._load_reference_clip(step_id)

    def _play_current_reference_clip(self):
        step_id=self._current_step_id()
        if step_id is not None:
            self._load_reference_clip(step_id)
            self.preview_player.play()

    def _load_reference_clip(self,step_id):
        step=self.step_lookup.get(step_id,{})
        clip_path=self._reference_clip_path(step_id)
        if clip_path.exists():
            self.preview_label.setText(f"Correct way: {step.get('name','')}")
            self.preview_player.setSource(QUrl.fromLocalFile(str(clip_path)))
        else:
            self.preview_label.setText(f"No clip yet for: {step.get('name','')}\nUse UPLOAD REFERENCE VIDEOS to add one.")
            self.preview_player.stop()

    def _on_repeat_step(self):
        step_id=self._current_step_id()
        if step_id is not None:
            self._load_reference_clip(step_id)
            self.preview_player.setPosition(0)
            self.preview_player.play()

    def closeEvent(self,event):
        if self.capture:
            self.capture.release()
        self.voice_alerts.shutdown()
        self.stream.close()
        event.accept()

def main():
    app=QApplication(sys.argv)
    window=MainWindowV3()
    window.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
