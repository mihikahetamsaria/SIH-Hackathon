from pathlib import Path
from PySide6.QtCore import Qt,QUrl
from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QListWidget,QListWidgetItem,QTableWidget,QTableWidgetItem,QHeaderView,QPushButton,QLabel,QGroupBox
from GUI import session_logger

class LogFilesDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Files & Recordings")
        self.resize(1000,700)
        layout=QHBoxLayout(self)
        left=QVBoxLayout()
        left.addWidget(QLabel("SESSIONS"))
        self.session_list=QListWidget()
        self.session_list.setFixedWidth(230)
        for path in session_logger.list_sessions():
            item=QListWidgetItem(path.stem.replace("session_",""))
            item.setData(Qt.UserRole,path)
            self.session_list.addItem(item)
        self.session_list.itemClicked.connect(self._load_session)
        left.addWidget(self.session_list,1)
        layout.addLayout(left)

        right=QVBoxLayout()
        self.session_info=QLabel("Select a session")
        self.session_info.setWordWrap(True)
        right.addWidget(self.session_info)
        self.table=QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(["Step","Status","Time","Confidence","Recording","Correction"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellDoubleClicked.connect(lambda row,_col:self._play_row(row))
        right.addWidget(self.table,1)
        buttons=QHBoxLayout()
        self.play_btn=QPushButton("PLAY EVENT RECORDING")
        self.play_btn.clicked.connect(lambda:self._play_row(self.table.currentRow()))
        self.full_btn=QPushButton("PLAY FULL SESSION")
        self.full_btn.clicked.connect(self._play_full_session)
        buttons.addWidget(self.play_btn)
        buttons.addWidget(self.full_btn)
        right.addLayout(buttons)
        self.preview_video=QVideoWidget()
        self.preview_video.setMinimumHeight(230)
        right.addWidget(self.preview_video)
        self.player=QMediaPlayer(self)
        self.audio=QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.preview_video)
        layout.addLayout(right,1)
        self._current_data={}
        self._current_events=[]
        self._current_session_video=None
        if self.session_list.count():
            self.session_list.setCurrentRow(0)
            self._load_session(self.session_list.item(0))
        else:
            self.session_info.setText("No sessions logged yet. Run an experiment first.")

    def _load_session(self,item):
        try:
            data=session_logger.load_session(item.data(Qt.UserRole))
        except Exception as e:
            self.session_info.setText(f"Could not read session: {e}")
            return
        self._current_data=data
        self._current_events=data.get("events") or data.get("event_recordings") or []
        self._current_session_video=data.get("session_video")
        status=data.get("run_status","unknown")
        recording="Available" if self._valid_path(self._current_session_video) else "Not found"
        observation=data.get("observation_log")
        self.session_info.setText(f"Status: {status}    |    Full session recording: {recording}\nObservation log: {observation or 'not recorded'}")
        self.table.setRowCount(len(self._current_events))
        steps=self._current_data.get("steps",[])
        for row,event in enumerate(self._current_events):
            step_id=event.get("step_id") or event.get("expected_step_id")
            step=next((s for s in steps if str(s.get("step_id"))==str(step_id)),{})
            timestamp=event.get("time_iso") or event.get("timestamp") or "-"
            confidence=event.get("confidence")
            conf=f"{float(confidence):.2f}" if isinstance(confidence,(int,float)) else "-"
            video=event.get("video_path") or ""
            correction=event.get("correction") or event.get("corrective_action") or ""
            self.table.setItem(row,0,QTableWidgetItem(step.get("name",f"Step {step_id}")))
            self.table.setItem(row,1,QTableWidgetItem(str(event.get("status",""))))
            self.table.setItem(row,2,QTableWidgetItem(str(timestamp)))
            self.table.setItem(row,3,QTableWidgetItem(conf))
            self.table.setItem(row,4,QTableWidgetItem("Available" if self._valid_path(video) else "-"))
            self.table.setItem(row,5,QTableWidgetItem(str(correction)))

    @staticmethod
    def _valid_path(value):
        return bool(value) and Path(str(value)).exists()

    def _play_row(self,row):
        if row<0 or row>=len(self._current_events):
            return
        video_path=self._current_events[row].get("video_path")
        if self._valid_path(video_path):
            self.player.setSource(QUrl.fromLocalFile(str(Path(video_path).resolve())))
            self.player.play()
        else:
            self.player.stop()
            self.session_info.setText(self.session_info.text().split("\n")[0]+"\nThis event has no saved recording.")

    def _play_full_session(self):
        if self._valid_path(self._current_session_video):
            self.player.setSource(QUrl.fromLocalFile(str(Path(self._current_session_video).resolve())))
            self.player.play()
        else:
            self.player.stop()
            self.session_info.setText(self.session_info.text().split("\n")[0]+"\nFull-session recording was not found.")
