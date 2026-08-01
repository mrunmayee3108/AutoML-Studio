from datetime import datetime 
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base
Base = declarative_base()
class ExperimentRecord(Base):
    __tablename__ = 'experiments'
    id = Column(Integer, primary_key=True, autoincrement = True)
    project_name = Column(String(100), nullable = False)
    task_type = Column(String(50), nullable = False)
    dataset_name = Column(String(200), nullable = False)
    best_model_name = Column(String(100), nullable = False)
    primary_metric_name = Column(String(50), nullable = False)
    primary_metric_value = Column(Float, nullable = False)
    version_id = Column(String(200), nullable = False, unique = True)
    report_path = Column(String(500), nullable = False)
    created_at = Column(DateTime, default = datetime.now)
    def to_dict(self):
        return{
            "id": self.id,
            "project_name": self.project_name,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "best_model": self.best_model_name,
            "metric": f"{self.primary_metric_name}: {self.primary_metric_value: .4f}",
            "version_id": self.version_id,
            "report_path": self.report_path,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }