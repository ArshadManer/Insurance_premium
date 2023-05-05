from src.train_model import train_model
from src.set_logger import set_logger
import logging


def main(config_path, stdout = False):
    set_logger('pipeline', stdout)

    logger = logging.getLogger('pipeline')
    try:
        model = train_model(config_path)
    except Exception as e:
        logger.info("Error")
        logger.error(e)


if __name__ == "__main__":
    import sys

    if "--help" in sys.argv:
        print("--config: to pass config file")
        sys.exit(0)

    if len(sys.argv) > 1:
        print("--help for help")
        sys.exit(0)

    config_file = "params.yaml"
    if "--config" in sys.argv:
        config_file = sys.argv[-1]

    stdout = False
    if "--stdout" in sys.argv:
        stdout = True

    main(config_file, stdout)
