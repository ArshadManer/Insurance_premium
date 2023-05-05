import joblib
import os
from src.config import get_config, get_base_path
from src.train_model import train_model
import logging

logger = logging.getLogger('load_model')


def load_model(config_path='params.yaml'):
    try:
        config = get_config(config_path)
        base_path = get_base_path()
        model_path = os.path.join(base_path, config['model_dirs'])
        model_path = os.path.join(model_path, config['webapp_model'])
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        logger.info('model not found')
        logger.info('training the new model')
        return train_model(config_path)
    except Exception as e:
        logger.error('Error while loading model')
        logger.exception(e)
        return None


if __name__ == '__main__':
    from src.set_logger import set_logger
    set_logger('test_load_model')
    model = load_model()
