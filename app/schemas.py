from pydantic import BaseModel, Field

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, description="Petal width (cm)")

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: list[float]
