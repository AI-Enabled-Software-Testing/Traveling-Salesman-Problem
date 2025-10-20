from pathlib import Path
import logging
from figure_scripts import time_budget_figures, relative_work_figures, time_budget_nn_figures, relative_work_nn_figures, box_plot_figures, random_baseline_figures
from constants import DATASET_FILENAME

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    logger.info("Generating TSP algorithm comparison figures...")
    logger.info(f"Output directory: {figures_dir.absolute()}")
    
    # Hard check for dataset (required for both scripts)
    dataset_path = Path("dataset") / DATASET_FILENAME
    if not dataset_path.exists():
        logger.error(f"Dataset missing ({dataset_path}).")
        logger.error(f"Run 'python setup_dataset.py' from root to download {DATASET_FILENAME} and other files.")
        logger.error("Figures cannot be generated without it.")
        return  # Exit without generating

    scripts = [
        ("box_plot_figures", box_plot_figures.main),
        ("random_baseline_figures", random_baseline_figures.main),
        ("time_budget_figures", time_budget_figures.main),
        ("time_budget_nn_figures", time_budget_nn_figures.main),
        ("relative_work_figures", relative_work_figures.main),
        ("relative_work_nn_figures", relative_work_nn_figures.main)
    ]
    
    for script_name, main_func in scripts:
        logger.info(f"Running {script_name}...")
        try:
            main_func()
            logger.info(f"Completed {script_name}")
        except Exception as e:
            logger.error(f"{script_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Generated figures in {figures_dir.absolute()}")


if __name__ == "__main__":
    main()
