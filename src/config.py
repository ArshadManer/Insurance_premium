import os
import yaml
import logging


def get_config(path: str) -> dict:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    logger = logging.getLogger('config')
    config = {}
    full_path = os.path.join(base_dir, path)
    if os.path.exists(full_path):
        logger.info(f"Loading configuration {path}")
        try:
            temp_config = yaml.safe_load(open(full_path, 'r'))
            config.update(temp_config)
        except FileNotFoundError as e:
            logger.error("File not found")
            logger.exception(e)
            raise FileNotFoundError
        except Exception as e:
            logger.error(path)
            logger.exception(e)
            raise Exception
    else:
        logger.info(f"Config {path} dose not exits at {full_path}")

    return config


def get_base_path() -> str:
    return os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    from src.set_logger import set_logger

    set_logger('test_config')
    config = get_config('params.yaml')
    print(config)
