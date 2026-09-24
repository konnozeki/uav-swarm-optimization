"""Launch the OpenGL/Pyglet UAV swarm simulator."""

import argparse
import multiprocessing


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=None,
        help="close automatically and print measured FPS after this many seconds",
    )
    parser.add_argument(
        "--single-buffer",
        action="store_true",
        help="experimental front-buffer mode; may not repaint under a compositor",
    )
    args = parser.parse_args()
    from src.simulator.pyglet_desktop import run_pyglet_simulator

    run_pyglet_simulator(
        benchmark_seconds=args.benchmark_seconds,
        double_buffer=not args.single_buffer,
    )
