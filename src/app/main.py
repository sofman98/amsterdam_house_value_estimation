from typing import Dict, Union
from fastapi import FastAPI
from src.utils.constants import DEFAULT_FEATURE_NAMES
from src.process.preprocessor import Preprocessor
from src.process.regressor import Regressor
import pandas as pd

app = FastAPI(title="WOZ Estimation API", description="API for WOZ value estimation")
preprocessor = Preprocessor.load_object("assets/preprocessors/preprocessor.joblib")
regressor = Regressor()


@app.on_event("startup")
def load_model() -> None:
    """
    Loads trained model.
    """
    regressor.load_model("assets/models/model.joblib")

    # return


@app.get("/api/get_woz_value")
def get_woz_value(
    single: Union[int, None] = 0,
    not_married_no_kids: Union[int, None] = 0,
    married_no_kids: Union[int, None] = 0,
    married_with_kids: Union[int, None] = 0,
    not_married_with_kids: Union[int, None] = 0,
    single_parent: Union[int, None] = 0,
    other: Union[int, None] = 0,
) -> Dict[str, int]:
    """
    Predicts and returns WOZ value. Example to copy:
      ?single=543&married_no_kids=37&not_married_no_kids=149&married_with_kids=14&not_married_with_kids=12&single_parent=22&other=12
    Args:
      - single (Union[int, None]): number of houses with single people in the area. Default to 0.
      - not_married_no_kids (Union[int, None]): number of houses with non-married people with no kids in the area. Default to 0.
      - married_no_kids (Union[int, None]): number of houses with married people with no kids in the area. Default to 0.
      - married_with_kids (Union[int, None]): number of houses with married people with kids in the area. Default to 0.
      - not_married_with_kids (Union[int, None]): number of houses with non-married people with kids in the area. Default to 0.
      - single_parent (Union[int, None]): number of houses with single parents in the area. Default to 0.
      - other (Union[int, None]): number of houses with people falling in other categories in the area. Default to 0.
    Returns:
      - (Dict[str, int]): Dictionnary with "woz_value" as key and the predicted woz as value.
    """

    # renaming the features to the default names
    data = {
        DEFAULT_FEATURE_NAMES["single"]: single,
        DEFAULT_FEATURE_NAMES["married_no_kids"]: married_no_kids,
        DEFAULT_FEATURE_NAMES["not_married_no_kids"]: not_married_no_kids,
        DEFAULT_FEATURE_NAMES["married_with_kids"]: married_with_kids,
        DEFAULT_FEATURE_NAMES["not_married_with_kids"]: not_married_with_kids,
        DEFAULT_FEATURE_NAMES["single_parent"]: single_parent,
        DEFAULT_FEATURE_NAMES["other"]: other,
    }
    # conclude the total number of houses
    data[DEFAULT_FEATURE_NAMES["total"]] = sum(data.values())

    # create DataFrame
    features = pd.DataFrame(data, index=[0])

    # normalize features
    scaled_features, _ = preprocessor.apply_normalization(features, None)

    # predict woz from features
    woz_value = regressor.predict(scaled_features, preprocessor)[0]

    return {"woz_value": int(woz_value)}
