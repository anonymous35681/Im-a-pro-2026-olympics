import importlib
import pkgutil

from loguru import logger

import graphs
from logger import setup_logger
from style import apply_global_style


def run_all_graphs() -> None:
    """Dynamically run all graphs in the src/graphs directory."""
    setup_logger()
    apply_global_style()

    logger.info("Starting graph generation process.")

    # Iterate over all modules in the src/graphs package
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(graphs.__path__):
        full_module_name = f"graphs.{module_name}"
        logger.info(f"Loading graph module: {full_module_name}")

        try:
            module = importlib.import_module(full_module_name)
            if not hasattr(module, "run"):
                logger.warning(f"No run() function found in module: {full_module_name}")
                continue

            logger.info(f"Executing run() function for: {full_module_name}")
            module.run()
            logger.info(f"Successfully generated graph for: {full_module_name}")
        except Exception as e:
            logger.error(f"Failed to execute module {full_module_name}: {e}")


if __name__ == "__main__":
    run_all_graphs()
