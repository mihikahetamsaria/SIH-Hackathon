from collections import defaultdict
from datetime import datetime
from PySide6.QtWidgets import QDialog,QVBoxLayout,QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QLabel
from GUI import session_logger


def _parse(ts):
    if ts is None:
        return None
    try:
        if isinstance(ts,(int,float)):
            return datetime.fromtimestamp(ts)
        return datetime.fromisoformat(str(ts))
    except (TypeError,ValueError,OverflowError):
        return None

class TaskTableDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analytics")
        self.resize(900,520)
        layout=QVBoxLayout(self)
        buckets=self._collect_buckets()
        if not any(buckets.values()):
            layout.addWidget(QLabel("No session history yet - run an experiment first."))
            return
        tabs=QTabWidget()
        tabs.addTab(self._build_table(buckets["hour"]),"Hourly")
        tabs.addTab(self._build_table(buckets["day"]),"Daily")
        tabs.addTab(self._build_table(buckets["week"]),"Weekly")
        layout.addWidget(tabs)

    def _collect_buckets(self):
        hour=defaultdict(lambda:[0,0,0,0,0,0])
        day=defaultdict(lambda:[0,0,0,0,0,0])
        week=defaultdict(lambda:[0,0,0,0,0,0])
        for path in session_logger.list_sessions():
            try:
                data=session_logger.load_session(path)
            except Exception:
                continue
            events=data.get("controller_events") or data.get("events") or []
            for event in events:
                status=str(event.get("status","")).upper()
                ts=_parse(event.get("time_iso") or event.get("timestamp"))
                if not ts:
                    continue
                idx={"CORRECT":0,"COMPLETE":0,"OUT_OF_ORDER":2,"WRONG":1,"UNEXPECTED":1,"IGNORED_REPEAT":3,"ABORTED":4}.get(status)
                if idx is None:
                    continue
                for bucket,key in ((hour,ts.strftime("%Y-%m-%d %H:00")),(day,ts.strftime("%Y-%m-%d")),(week,f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}")):
                    bucket[key][idx]+=1
        return {"hour":hour,"day":day,"week":week}

    def _build_table(self,bucket):
        table=QTableWidget(len(bucket),7)
        table.setHorizontalHeaderLabels(["Period","Correct","Wrong / Unexpected","Out of order","Repeated complete","Aborted","Total events"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row,(key,counts) in enumerate(sorted(bucket.items())):
            table.setItem(row,0,QTableWidgetItem(key))
            table.setItem(row,1,QTableWidgetItem(str(counts[0]+counts[1])))
            table.setItem(row,2,QTableWidgetItem(str(counts[1])))
            table.setItem(row,3,QTableWidgetItem(str(counts[2])))
            table.setItem(row,4,QTableWidgetItem(str(counts[3])))
            table.setItem(row,5,QTableWidgetItem(str(counts[4])))
            table.setItem(row,6,QTableWidgetItem(str(sum(counts))))
        return table
