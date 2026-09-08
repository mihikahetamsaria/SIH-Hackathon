"""
"Task Table" toolbar dialog: aggregates every saved session archive
(logs/session_*.json) into Hourly / Daily / Weekly tabs, so you can see
progress across multiple runs, not just the one currently live.
"""

from collections import defaultdict
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
)

import session_logger

STATUS_COLUMN = {"done": 0, "skipped": 1, "out_of_order": 2}


def _parse(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


class TaskTableDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Task Table")
        self.resize(720, 500)

        layout = QVBoxLayout(self)
        buckets = self._collect_buckets()

        if not any(buckets.values()):
            layout.addWidget(
                QLabel("No session history yet - run the app a few times and this fills in.")
            )
            return

        tabs = QTabWidget()
        tabs.addTab(self._build_table(buckets["hour"]), "Hourly")
        tabs.addTab(self._build_table(buckets["day"]), "Daily")
        tabs.addTab(self._build_table(buckets["week"]), "Weekly")
        layout.addWidget(tabs)

    def _collect_buckets(self):
        hour = defaultdict(lambda: [0, 0, 0])
        day = defaultdict(lambda: [0, 0, 0])
        week = defaultdict(lambda: [0, 0, 0])

        for path in session_logger.list_sessions():
            data = session_logger.load_session(path)
            for s in data.get("steps", []):
                idx = STATUS_COLUMN.get(s.get("status"))
                if idx is None:
                    continue
                ts = _parse(s.get("ts_end") or s.get("ts_start"))
                if not ts:
                    continue
                hour[ts.strftime("%Y-%m-%d %H:00")][idx] += 1
                day[ts.strftime("%Y-%m-%d")][idx] += 1
                iso_year, iso_week, _ = ts.isocalendar()
                week[f"{iso_year}-W{iso_week:02d}"][idx] += 1

        return {"hour": hour, "day": day, "week": week}

    def _build_table(self, bucket: dict) -> QTableWidget:
        table = QTableWidget(len(bucket), 4)
        table.setHorizontalHeaderLabels(["Period", "Done", "Skipped", "Out of order"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, (key, counts) in enumerate(sorted(bucket.items())):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(str(counts[0])))
            table.setItem(row, 2, QTableWidgetItem(str(counts[1])))
            table.setItem(row, 3, QTableWidgetItem(str(counts[2])))
        return table
