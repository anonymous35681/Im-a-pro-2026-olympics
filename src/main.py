import importlib
import os
import pkgutil
import sys

from loguru import logger

import graphs
from logger import setup_logger
from style import apply_global_style


def run_single_graph(graph_number: str) -> None:
    """Run a specific graph by number."""
    setup_logger()
    apply_global_style()

    module_name = f"graph{graph_number}"
    full_module_name = f"graphs.{module_name}"

    logger.info(f"Loading graph module: {full_module_name}")

    try:
        module = importlib.import_module(full_module_name)
        if not hasattr(module, "run"):
            logger.error(f"No run() function found in module: {full_module_name}")
            sys.exit(1)

        logger.info(f"Executing run() function for: {full_module_name}")
        module.run()
        logger.success(f"Successfully generated graph for: {full_module_name}")

    except ImportError:
        logger.error(f"Module not found: {full_module_name}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to execute module {full_module_name}: {e}")
        sys.exit(1)


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
    # Check if a graph number was provided
    if len(sys.argv) == 2:
        graph_number: str = sys.argv[1]
        run_single_graph(graph_number)
        sys.exit(os.EX_OK)

    if len(sys.argv) > 2:
        logger.error("Too many args")
        sys.exit(os.EX_USAGE)

    run_all_graphs()
