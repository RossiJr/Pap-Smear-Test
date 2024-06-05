import numpy as np
import xgboost
from django.conf import settings
import operations.image_manipulation as im


def classify_image(image: np.ndarray, str_model: str):
    if image is None or str_model is None:
        return None
    pre_proc = im.pre_process_image(image)
    img_dmatriz = xgboost.DMatrix(pre_proc)

    if str_model == 'xgboost_binary':
        binary_prediction = settings.XGBOOST_MODELS['binary'].predict(img_dmatriz)
        binary_prediction_proba = settings.XGBOOST_MODELS['binary'].predict_proba(img_dmatriz)
        return binary_prediction, binary_prediction_proba
    elif str_model == 'xgboost_multiclass':
        multiclass_prediction = settings.XGBOOST_MODELS['multiclass'].predict(img_dmatriz)
        multiclass_prediction_proba = settings.XGBOOST_MODELS['multiclass'].predict_proba(img_dmatriz)
        return multiclass_prediction, multiclass_prediction_proba
