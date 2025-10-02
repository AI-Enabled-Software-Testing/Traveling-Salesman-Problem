import random
import logging
from constants import RANDOM_SEED

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("TSP Solver")
    # Setup
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    main()
