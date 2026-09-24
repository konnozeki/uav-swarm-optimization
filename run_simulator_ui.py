"""Launch the native Python desktop simulator."""

import multiprocessing

from src.simulator.desktop import run_desktop_simulator


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_desktop_simulator()
