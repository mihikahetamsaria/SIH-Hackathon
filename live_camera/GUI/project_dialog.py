from pathlib import Path
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QListWidget,QListWidgetItem,QPushButton,QMessageBox,QInputDialog,QFileDialog

class ProjectsDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.parent_window=parent
        self.setWindowTitle("Projects")
        self.resize(760,520)
        root=QVBoxLayout(self)
        title=QLabel("EXPERIMENT PROJECTS")
        title.setStyleSheet("font-size:20px;font-weight:800;")
        root.addWidget(title)
        current=getattr(parent,"current_experiment_name",None)
        current_path=getattr(parent,"current_experiment_path",None)
        status=QLabel(f"Current: {current or 'No experiment loaded'}"+(f"\nFile: {current_path}" if current_path else ""))
        status.setWordWrap(True)
        root.addWidget(status)
        self.list=QListWidget()
        root.addWidget(self.list,1)
        self._populate()
        buttons=QHBoxLayout()
        load=QPushButton("LOAD SELECTED")
        load.clicked.connect(self._load_selected)
        rename=QPushButton("RENAME FILE")
        rename.clicked.connect(self._rename_selected)
        refs=QPushButton("REFERENCE VIDEOS")
        refs.clicked.connect(self._manage_references)
        delete=QPushButton("DELETE")
        delete.clicked.connect(self._delete_selected)
        refresh=QPushButton("REFRESH")
        refresh.clicked.connect(self._populate)
        close=QPushButton("CLOSE")
        close.clicked.connect(self.close)
        buttons.addWidget(load); buttons.addWidget(rename); buttons.addWidget(refs); buttons.addWidget(delete); buttons.addWidget(refresh); buttons.addStretch(); buttons.addWidget(close)
        root.addLayout(buttons)
        self.list.itemDoubleClicked.connect(lambda _: self._load_selected())

    def _populate(self):
        self.list.clear()
        data_dir=Path(__file__).parent/"data"
        roots=[data_dir/"projects",data_dir]
        paths=[]; seen=set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.json"):
                if path.name=="step_record_log.json" or path.resolve() in seen:
                    continue
                try:
                    definition=json.loads(path.read_text(encoding="utf-8"))
                    steps=definition.get("steps")
                    if not isinstance(steps,list) or not steps:
                        continue
                    seen.add(path.resolve()); paths.append(path)
                    name=definition.get("experiment_name",path.stem)
                    item=QListWidgetItem(f"{name}   •   {len(steps)} steps\n{path}")
                    item.setData(Qt.UserRole,str(path))
                    self.list.addItem(item)
                except Exception:
                    continue
        if self.list.count()==0:
            self.list.addItem("No saved projects found.")

    def _selected_path(self):
        item=self.list.currentItem()
        if not item:
            QMessageBox.information(self,"Projects","Select a project first.")
            return None
        path=item.data(Qt.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self,"Projects","Select a valid saved project.")
            return None
        return Path(path)

    def _rename_selected(self):
        path=self._selected_path()
        if path is None:
            return
        old_stem=path.stem
        new_name,ok=QInputDialog.getText(self,"Rename Project File","New file name:",text=old_stem)
        if not ok:
            return
        new_name=new_name.strip()
        if not new_name:
            return
        if not new_name.lower().endswith(".json"):
            new_name += ".json"
        target=path.with_name(new_name)
        if target.resolve()!=path.resolve() and target.exists():
            QMessageBox.warning(self,"Projects",f"A project named '{target.name}' already exists.")
            return
        try:
            old_project_dir=path.parent / path.stem
            new_project_dir=target.parent / target.stem
            if old_project_dir.exists() and old_project_dir.is_dir() and old_project_dir.resolve()!=new_project_dir.resolve():
                if new_project_dir.exists():
                    QMessageBox.warning(self,"Projects",f"A project data folder named '{new_project_dir.name}' already exists.")
                    return
                old_project_dir.rename(new_project_dir)
            path.rename(target)
            QMessageBox.information(self,"Projects","Project file and its reference-video folder were renamed successfully.")
            self._populate()
        except Exception as e:
            QMessageBox.critical(self,"Projects",f"Could not rename project:\n{e}")

    def _delete_selected(self):
        path=self._selected_path()
        if path is None:
            return
        if QMessageBox.question(self,"Delete Project",f"Delete '{path.name}'?\n\nThis deletes the experiment JSON. Reference videos stored separately are not deleted.",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:
            return
        try:
            path.unlink()
            QMessageBox.information(self,"Projects","Project deleted successfully.")
            self._populate()
        except Exception as e:
            QMessageBox.critical(self,"Projects",f"Could not delete project:\n{e}")

    def _manage_references(self):
        path=self._selected_path()
        if path is None:
            return
        try:
            definition=json.loads(path.read_text(encoding="utf-8"))
            steps=definition.get("steps",[])
            if not isinstance(steps,list) or not steps:
                QMessageBox.warning(self,"Reference Videos","This project has no valid steps.")
                return
            folder=path.parent / path.stem / "reference_clips"
            folder.mkdir(parents=True,exist_ok=True)
            dlg=QDialog(self)
            dlg.setWindowTitle(f"Reference Videos — {definition.get('experiment_name',path.stem)}")
            dlg.resize(720,460)
            root=QVBoxLayout(dlg)
            root.addWidget(QLabel("Manage the reference video associated with each experiment step."))
            lst=QListWidget()
            root.addWidget(lst,1)
            def populate():
                lst.clear()
                for step in steps:
                    sid=int(step.get("step_id"))
                    name=step.get("name",f"Step {sid}")
                    matches=list(folder.glob(f"step_{sid}.*"))
                    if matches:
                        text=f"Step {sid}: {name}  •  {matches[0].name}"
                    else:
                        text=f"Step {sid}: {name}  •  No reference video"
                    item=QListWidgetItem(text)
                    item.setData(Qt.UserRole,(sid,name))
                    lst.addItem(item)
            populate()
            row=QHBoxLayout()
            replace=QPushButton("ADD / REPLACE")
            remove=QPushButton("REMOVE VIDEO")
            close=QPushButton("CLOSE")
            row.addWidget(replace); row.addWidget(remove); row.addStretch(); row.addWidget(close)
            root.addLayout(row)
            def selected():
                item=lst.currentItem()
                if not item:
                    QMessageBox.information(dlg,"Reference Videos","Select a step first.")
                    return None
                return item.data(Qt.UserRole)
            def replace_video():
                data=selected()
                if data is None:
                    return
                sid,name=data
                source,_=QFileDialog.getOpenFileName(dlg,f"Reference video — Step {sid}: {name}",str(folder),"Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
                if not source:
                    return
                for old in folder.glob(f"step_{sid}.*"):
                    try: old.unlink()
                    except OSError: pass
                ext=Path(source).suffix.lower() or ".mp4"
                target=folder/f"step_{sid}{ext}"
                try:
                    target.write_bytes(Path(source).read_bytes())
                    populate()
                except Exception as e:
                    QMessageBox.critical(dlg,"Reference Videos",f"Could not save reference video:\n{e}")
            def remove_video():
                data=selected()
                if data is None:
                    return
                sid,name=data
                matches=list(folder.glob(f"step_{sid}.*"))
                if not matches:
                    return
                if QMessageBox.question(dlg,"Remove Reference Video",f"Remove the reference video for Step {sid}: {name}?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:
                    return
                for old in matches:
                    try: old.unlink()
                    except OSError: pass
                populate()
            replace.clicked.connect(replace_video)
            remove.clicked.connect(remove_video)
            close.clicked.connect(dlg.accept)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self,"Reference Videos",f"Could not open reference videos:\n{e}")

    def _load_selected(self):
        item=self.list.currentItem()
        if not item:
            return
        path=item.data(Qt.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self,"Projects","Select a valid saved project.")
            return
        self.close()
        self.parent_window._load_experiment_json(path=path,saved_project=True)
