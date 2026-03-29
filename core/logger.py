import logging
import os

def get_logger():

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("rag_pipeline")

    if not logger.handlers:

        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler("logs/pipeline.log")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger