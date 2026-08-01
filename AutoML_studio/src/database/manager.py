import os 
import logging 
from typing import List, Dict, Any, Optional 
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from src.database.models import Base, ExperimentRecord
logging.basicConfig(level=logging.INFO, format = "%(asctime)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
class DatabaseManager:
    # handles connections n crud operations for the sqlite db using sqlalchemy sessions.
    def __init__(self, db_path: str="sqlite:///automl.db"):
        self.engine = create_engine(db_path, connect_args={"check_same_thread":False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"DatabaseManager initialized. Connected to DB: {db_path}")

    def log_experiment(self, experiment_data: Dict[str, Any]) -> Optional[int]:
        # inserts a new completed automl experiment into the database.
        session = self.SessionLocal()
        try:
            new_record = ExperimentRecord(
               project_name=experiment_data.get("project_name", "Untitled"),
                task_type=experiment_data.get("task_type", "unknown"),
                dataset_name=experiment_data.get("dataset_name", "unknown.csv"),
                best_model_name=experiment_data.get("best_model_name", "Unknown"),
                primary_metric_name=experiment_data.get("primary_metric_name", "Score"),
                primary_metric_value=experiment_data.get("primary_metric_value", 0.0),
                version_id=experiment_data.get("version_id", ""),
                report_path=experiment_data.get("report_path", "") 
            )
            session.add(new_record)
            session.commit()
            session.refresh(new_record)
            logger.info(f"Experiment successfully logged to database with ID: {new_record.id}")
            return new_record.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to log experiment to database: {str(e)}")
            return None
        finally:
            session.close()

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Retrieves the history of all experiments, sorted newest first.
        Useful for populating Streamlit dashboard tables.
        """
        session = self.SessionLocal()
        try:
            # Query all records, ordered by creation time descending
            records = session.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).all()
            return [record.to_dict() for record in records]
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve experiments: {str(e)}")
            return []
        finally:
            session.close()