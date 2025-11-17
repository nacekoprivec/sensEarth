import enum
from pydantic import BaseModel, constr, Field
from typing import Optional


class AvailableConfigs(enum.Enum):
    BorderCheck = "border_check.json"
    Clustering = "clustering.json"
    Cumulative = "cumulative.json"
    EMAPercentile = "ema_percentile.json"
    EMA = "ema.json"
    Filtering = "filtering.json"
    GAN = "gan.json"
    Hampel = "hampel.json"
    IsolationForest = "isolation_forest.json"
    LinearFit = "linear_fit.json"
    MACD = "macd.json"
    PCA = "pca.json"
    RRCF = "rrcf_trees.json"
    TrendClassification = "trend_classification.json"
    Welford = "welford.json"

class ConfigNameModel(BaseModel):
    name: str = Field(..., max_length=10, pattern=r'^[A-Za-z]{3}[A-Za-z0-9_]{0,7}$') 

class DetectorCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config_name: Optional[str] = None
    config : Optional[str] = None
    anomaly_detection_alg: Optional[list] = None
    anomaly_detection_conf: Optional[list] = None

class DetectorUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class StatusUpdate(BaseModel):
    status: str