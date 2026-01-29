from llava.scripts.utils import setup_logger

def test_logger_creation():
    logger = setup_logger(log_file='/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/logs/test.log')
    assert logger is not None