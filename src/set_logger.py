import logging
import os
import yaml
import sys


def set_logger(file_name, stdout=False):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config = yaml.safe_load(open(os.path.join(base_dir, 'params.yaml'), 'r'))
    log_path = config.get('log_dir')
    base_dir = os.path.join(base_dir, log_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    root = logging.getLogger()
    root.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)10s - %(levelname)7s - %(message)s')
    handlers = [logging.FileHandler(filename=f"{base_dir}/{file_name}.log")]

    if stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(logging.INFO)
    root.info("++++++++++++++++++++++++++++++Insurance-Price-Prediction+++++++++++++++++++++++++++++++")
    root.info(f"logging file location: {file_name}")


if __name__ == '__main__':
    set_logger('test_logs')
    logger = logging.getLogger("test")
    logger.info("test logging")
