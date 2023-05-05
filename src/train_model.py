from src.split_data import split_data
from src.config import get_config, get_base_path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import os
import joblib
import logging
import json

logger = logging.getLogger('train_model')


def eval_metrics(act, pred):
    r2score = r2_score(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))
    mse = mean_squared_error(act, pred)
    mae = mean_absolute_error(act, pred)
    return r2score, rmse, mse, mae


def train_model(config_path: str = 'params.yaml'):
    config = get_config(config_path)
    base_path = get_base_path()
    train, test = split_data(config_path)
    if train is not None and test is not None:
        model_dir = config['model_dirs']
        model_dir = os.path.join(base_path, model_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        if os.path.exists(model_path):
            logger.info("model already exits")
            return joblib.load(model_path)

        logger.info('training the model')
        target_col = config['target_data']
        model_config = config['model']

        x_train, x_test = train.drop(target_col, axis=1), test.drop(target_col, axis=1)
        y_train, y_test = train[target_col], test[target_col]

        logger.info('initializing the model')
        gb = GradientBoostingRegressor(**model_config)

        logger.info('training the model')
        gb.fit(x_train, y_train)

        logger.info('training completed')
        logger.info('testing the model')
        y_pred = gb.predict(x_test)
        (r2, rmse, mae, mse) = eval_metrics(y_test, y_pred)
        logger.info('test result')
        logger.info(f'R2: {r2 * 100} %, RMSE: {rmse}, MAE: {mae}, MSE: {mse}')

        logger.info("saving the model")
        joblib.dump(gb, model_path)
        logger.info("model saved")

        #################reports logging###############

        os.makedirs(os.path.join(base_path, 'reports'), exist_ok=True)

        scores_file = os.path.join(base_path, config["reports"]["scores"])
        params_file = os.path.join(base_path, config["reports"]["params"])

        with open(scores_file, "w+") as f:
            scores = {
                "rmse": rmse,
                "mse": mse,
                "r2 score": r2,
                "rmse": rmse,
            }
            json.dump(scores, f, indent=4)
        with open(params_file, "w+") as f:
            json.dump(model_config, f, indent=4)

        return gb

    else:
        raise Exception('Train or test data not received')


if __name__ == '__main__':
    from src.set_logger import set_logger

    set_logger('train_model')
    train_model()
