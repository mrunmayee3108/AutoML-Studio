import pandas as pd
import logging 
import io
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filename: str = ""
    
    def load_file(self, file_buffer: io.BytesIO, filename: str) -> bool:
        self.filename = filename
        extension = filename.split('.')[-1].lower()

        try:
            if extension == 'csv':
                self.df = pd.read_csv(file_buffer)
            elif extension in ['xls', 'xlsx']:
                self.df = pd.read_excel(file_buffer, engine='openpyxl')
            else:
                logger.error(f"Unsupported format: {extension}")
                return False
            if self.df.empty:
                logger.warning("Uploaded file is empty")
                return False
            logger.info(f"Loaded {filename} with shape {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        # dataset shape, size and missing vals count.
        if self.df is None: 
            return {}
        memory_mb = self.df.memory_usage(deep=True).sum()/(1024*1024)

        return{
            "filename": self.filename,
            "shape": self.df.shape,
            "memory_usage_mb": round(memory_mb, 2),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "total_missing": int(self.df.isnull().sum()),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_per_column": self.df.isnull().sum().to_dict()
        }

