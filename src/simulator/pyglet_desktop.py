from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
from typing import Callable

import networkx as nx
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.window import key, mouse

from ..graph_ops import communication_graph_from_positions
from ..obstacles import AxisAlignedRectangle
from .city_map import CityMap, generate_city_map, load_city_map, save_city_map
from .engine import ALGORITHM_LABELS, execute_simulation


WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 880
SIDEBAR_WIDTH = 330
MAP_PADDING = 18
DEFAULT_MAP_PATH = Path("data/simulator_map.json")

BG = (15, 20, 29, 255)
PANEL = (24, 31, 43, 255)
MAP_BG = (12, 17, 25, 255)
GRID = (40, 50, 65, 130)
TEXT = (226, 232, 240, 255)
MUTED = (145, 155, 170, 255)
BLUE = (68, 164, 255, 255)
GREEN = (82, 224, 145, 255)
AMBER = (245, 185, 52, 255)
RED = (235, 111, 92, 255)


class Button:
    def __init__(
        self,
        batch: pyglet.graphics.Batch,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        callback: Callable[[], None],
        *,
        primary: bool = False,
    ) -> None:
        self.x, self.y, self.width, self.height = x, y, width, height
        self.callback = callback
        self.enabled = True
        self.base_color = (38, 91, 145) if primary else (42, 52, 68)
        self.hover_color = (49, 119, 188) if primary else (56, 70, 91)
        self.pressed_color = (31, 75, 119) if primary else (32, 40, 54)
        self.pressed = False
        self.background = shapes.Rectangle(
            x, y, width, height, color=self.base_color, batch=batch
        )
        self.label = pyglet.text.Label(
            text,
            x=x + width / 2,
            y=y + height / 2,
            anchor_x="center",
            anchor_y="center",
            font_name="DejaVu Sans",
            font_size=10,
            color=TEXT,
            batch=batch,
        )

    @property
    def text(self) -> str:
        return self.label.text

    @text.setter
    def text(self, value: str) -> None:
        self.label.text = value

    def hit(self, x: float, y: float) -> bool:
        return (
            self.enabled
            and self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )

    def press(self) -> None:
        if self.enabled:
            self.callback()

    def set_hovered(self, hovered: bool) -> None:
        if not self.enabled:
            return
        self.background.color = (
            self.pressed_color
            if self.pressed
            else self.hover_color if hovered else self.base_color
        )

    def set_pressed(self, pressed: bool) -> None:
        self.pressed = pressed and self.enabled
        self.background.color = self.pressed_color if self.pressed else self.base_color

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.pressed = False
        self.background.color = self.base_color
        self.background.opacity = 255 if enabled else 90
        self.label.opacity = 255 if enabled else 110


class ValueSlider:
    def __init__(
        self,
        batch: pyglet.graphics.Batch,
        x: float,
        y: float,
        width: float,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        callback: Callable[[float], None] | None = None,
        *,
        integer: bool = False,
        unit: str = "",
        live: bool = False,
    ) -> None:
        self.x, self.y, self.width = x, y, width
        self.minimum, self.maximum = minimum, maximum
        self._value = value
        self.callback = callback
        self.integer = integer
        self.unit = unit
        self.live = live
        self.dragging = False
        self.name = pyglet.text.Label(
            label,
            x=x,
            y=y + 16,
            font_name="DejaVu Sans",
            font_size=9,
            color=MUTED,
            batch=batch,
        )
        self.value_label = pyglet.text.Label(
            "",
            x=x + width,
            y=y + 16,
            anchor_x="right",
            font_name="DejaVu Sans",
            font_size=9,
            color=TEXT,
            batch=batch,
        )
        self.track = shapes.Rectangle(x, y + 2, width, 4, color=(55, 67, 84), batch=batch)
        self.fill = shapes.Rectangle(x, y + 2, 1, 4, color=BLUE, batch=batch)
        self.knob = shapes.Circle(x, y + 4, 7, segments=16, color=TEXT, batch=batch)
        self._refresh()

    @property
    def value(self) -> float:
        return round(self._value) if self.integer else self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = min(max(float(value), self.minimum), self.maximum)
        self._refresh()

    def _refresh(self) -> None:
        ratio = (self._value - self.minimum) / max(self.maximum - self.minimum, 1e-12)
        self.fill.width = max(1, self.width * ratio)
        self.knob.x = self.x + self.width * ratio
        shown = str(int(round(self._value))) if self.integer else f"{self._value:.1f}"
        self.value_label.text = f"{shown}{self.unit}"

    def hit(self, x: float, y: float) -> bool:
        return self.x - 8 <= x <= self.x + self.width + 8 and self.y - 7 <= y <= self.y + 14

    def set_from_x(self, x: float) -> None:
        ratio = min(max((x - self.x) / self.width, 0.0), 1.0)
        value = self.minimum + ratio * (self.maximum - self.minimum)
        self._value = round(value) if self.integer else value
        self._refresh()
        if self.live and self.callback:
            self.callback(self.value)

    def commit(self) -> None:
        if not self.live and self.callback:
            self.callback(self.value)


class PygletSwarmSimulator(pyglet.window.Window):
    """Native OpenGL simulator using persistent, batched geometry."""

    def __init__(
        self,
        *,
        double_buffer: bool = True,
    ) -> None:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        width = min(WINDOW_WIDTH, max(1100, screen.width - 40))
        height = min(WINDOW_HEIGHT, max(700, screen.height - 80))
        config = pyglet.gl.Config(
            double_buffer=double_buffer,
            sample_buffers=1,
            samples=4,
        )
        try:
            super().__init__(
                width=width,
                height=height,
                caption="UAV Swarm Simulator — Pyglet/OpenGL",
                resizable=False,
                vsync=False,
                config=config,
            )
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=double_buffer)
            super().__init__(
                width=width,
                height=height,
                caption="UAV Swarm Simulator — Pyglet/OpenGL",
                resizable=False,
                vsync=False,
                config=config,
            )
        self.set_vsync(False)
        pyglet.gl.glClearColor(*(channel / 255 for channel in BG))

        self.city_map = generate_city_map()
        self.settings = {
            "n_uavs": 6,
            "sensing_radius": 175.0,
            "communication_radius": 280.0,
            "min_separation": 3.0,
            "max_speed": 40.0,
            "obstacle_clearance": 8.0,
            "algorithm": "transition-aware",
            "seed": 0,
            "population": 18,
            "generations": 20,
        }
        self.target_count = 80
        self.obstacle_count = 10
        self.map_seed = 42
        self.difficulty_index = 1
        self.algorithm_index = list(ALGORITHM_LABELS).index("transition-aware")
        self.edit_modes = ["Xem", "Thêm mục tiêu", "Thêm vật cản", "Xóa"]
        self.edit_mode_index = 0
        self.show_sensing = True
        self.show_links = True
        self.show_trails = True
        self.playing = False
        self.playback_speed = 1.0
        self.simulation_time = 0.0
        self.result: dict | None = None
        self.future: Future | None = None
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )

        self.static_batch = pyglet.graphics.Batch()
        self.trail_batch = pyglet.graphics.Batch()
        self.dynamic_batch = pyglet.graphics.Batch()
        self.ui_batch = pyglet.graphics.Batch()
        self.buttons: list[Button] = []
        self.sliders: list[ValueSlider] = []
        self.active_slider: ValueSlider | None = None
        self.active_button: Button | None = None
        self.target_shapes: list[shapes.Circle] = []
        self.sensing_shapes: list[shapes.Circle] = []
        self.link_shapes: dict[tuple[int, int], shapes.Line] = {}
        self.uav_shapes: list[shapes.Circle] = []
        self.uav_labels: list[pyglet.text.Label] = []
        self.trail_shapes: list[shapes.Line] = []
        self.last_trail_index = -1
        self.last_covered: np.ndarray | None = None
        self.dynamic_uav_count = 0
        self.draw_ms = 0.0
        self.present_ms = 0.0
        self.frame_count = 0
        self.fps_started = time.perf_counter()
        self.latest_fps = 0.0
        self.status = "Sẵn sàng — nhấn CHẠY TỐI ƯU"

        self._build_ui()
        self._rebuild_static()
        self._update_dynamic(force=True)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    @property
    def map_rect(self) -> tuple[float, float, float, float]:
        x = 18.0
        y = 75.0
        width = self.width - SIDEBAR_WIDTH - 54.0
        height = self.height - 112.0
        return x, y, width, height

    def _label(self, text: str, x: float, y: float, *, size: int = 10,
               color=TEXT, bold: bool = False, width: int | None = None):
        return pyglet.text.Label(
            text,
            x=x,
            y=y,
            width=width,
            multiline=width is not None,
            font_name="DejaVu Sans",
            font_size=size,
            weight="bold" if bold else "normal",
            color=color,
            batch=self.ui_batch,
        )

    def _add_button(self, x, y, width, text, callback, *, primary=False) -> Button:
        button = Button(self.ui_batch, x, y, width, 29, text, callback, primary=primary)
        self.buttons.append(button)
        return button

    def _add_slider(self, x, y, width, label, minimum, maximum, value,
                    callback=None, *, integer=False, unit="", live=False) -> ValueSlider:
        slider = ValueSlider(
            self.ui_batch, x, y, width, label, minimum, maximum, value,
            callback, integer=integer, unit=unit, live=live,
        )
        self.sliders.append(slider)
        return slider

    def _build_ui(self) -> None:
        side_x = self.width - SIDEBAR_WIDTH - 18
        shapes.Rectangle(
            side_x - 12, 18, SIDEBAR_WIDTH + 12, self.height - 36,
            color=PANEL, batch=self.ui_batch,
        )
        self._label("UAV SWARM SIMULATOR", 18, self.height - 31, size=15, color=BLUE, bold=True)
        self._label("Pyglet + OpenGL theo lô", 250, self.height - 29, size=10, color=MUTED)
        self.fps_label = self._label("FPS: —", self.width - 420, self.height - 31, size=11, color=GREEN)

        x = side_x
        y = self.height - 48
        self._label("CẤU HÌNH", x, y, size=13, color=BLUE, bold=True)
        y -= 42
        self.difficulty_button = self._add_button(x, y, 145, "Map: Trung bình", self._cycle_difficulty)
        self.edit_button = self._add_button(x + 155, y, 145, "Sửa: Xem", self._cycle_edit_mode)
        y -= 43
        self.target_slider = self._add_slider(
            x, y, 140, "Mục tiêu", 10, 250, self.target_count,
            lambda value: setattr(self, "target_count", int(value)), integer=True,
        )
        self.obstacle_slider = self._add_slider(
            x + 160, y, 140, "Vật cản", 0, 30, self.obstacle_count,
            lambda value: setattr(self, "obstacle_count", int(value)), integer=True,
        )
        y -= 50
        self.uav_slider = self._add_slider(
            x, y, 300, "Số UAV", 2, 20, self.settings["n_uavs"],
            self._set_uav_count, integer=True,
        )
        y -= 50
        self.sensing_slider = self._add_slider(
            x, y, 300, "Bán kính cảm biến", 50, 400,
            self.settings["sensing_radius"], self._set_sensing, unit=" m",
        )
        y -= 50
        self.communication_slider = self._add_slider(
            x, y, 300, "Bán kính liên lạc", 80, 600,
            self.settings["communication_radius"], self._set_communication, unit=" m",
        )
        y -= 43
        self._add_button(x, y, 95, "Sinh map", self._generate_map)
        self._add_button(x + 102, y, 95, "Lưu map", self._save_map)
        self._add_button(x + 204, y, 96, "Mở map", self._load_map)

        y -= 47
        self._label("THUẬT TOÁN", x, y, size=11, color=BLUE, bold=True)
        y -= 35
        self.algorithm_button = self._add_button(
            x, y, 300, self._algorithm_button_text(), self._cycle_algorithm,
        )
        y -= 50
        self.population_slider = self._add_slider(
            x, y, 140, "Quần thể", 4, 80, self.settings["population"],
            lambda value: self.settings.__setitem__("population", int(value)), integer=True,
        )
        self.generation_slider = self._add_slider(
            x + 160, y, 140, "Số thế hệ", 1, 100, self.settings["generations"],
            lambda value: self.settings.__setitem__("generations", int(value)), integer=True,
        )
        y -= 46
        self.run_button = self._add_button(
            x, y, 300, "CHẠY TỐI ƯU", self._start_solver, primary=True,
        )

        y -= 49
        self._label("HIỂN THỊ", x, y, size=11, color=BLUE, bold=True)
        y -= 34
        self.sensing_button = self._add_button(x, y, 95, "Cảm biến: Bật", self._toggle_sensing)
        self.links_button = self._add_button(x + 102, y, 95, "Liên kết: Bật", self._toggle_links)
        self.trails_button = self._add_button(x + 204, y, 96, "Vệt bay: Bật", self._toggle_trails)
        y -= 41
        self.play_button = self._add_button(x, y, 95, "Phát", self._toggle_play)
        self._add_button(x + 102, y, 95, "Về đầu", self._rewind)
        self.speed_button = self._add_button(x + 204, y, 96, "Tốc độ: 1x", self._cycle_speed)

        y -= 48
        self._label("KẾT QUẢ", x, y, size=11, color=BLUE, bold=True)
        y -= 25
        self.metrics_label = self._label(
            "Chưa có kết quả", x, y, size=9, color=TEXT, width=300,
        )
        self.metrics_label.anchor_y = "top"
        self.metrics_label.height = 115

        self.status_label = self._label(
            self.status, 18, 48, size=10, color=MUTED,
            width=int(self.map_rect[2]),
        )
        self.timeline = self._add_slider(
            18, 20, self.map_rect[2], "Thời gian", 0.0, 1.0, 0.0,
            self._seek, unit=" s", live=True,
        )

    def _algorithm_button_text(self) -> str:
        key_name = list(ALGORITHM_LABELS)[self.algorithm_index]
        return ALGORITHM_LABELS[key_name]

    def _cycle_difficulty(self) -> None:
        difficulties = ["Dễ", "Trung bình", "Khó"]
        self.difficulty_index = (self.difficulty_index + 1) % len(difficulties)
        self.difficulty_button.text = f"Map: {difficulties[self.difficulty_index]}"

    def _cycle_edit_mode(self) -> None:
        self.edit_mode_index = (self.edit_mode_index + 1) % len(self.edit_modes)
        self.edit_button.text = f"Sửa: {self.edit_modes[self.edit_mode_index]}"

    def _cycle_algorithm(self) -> None:
        self.algorithm_index = (self.algorithm_index + 1) % len(ALGORITHM_LABELS)
        key_name = list(ALGORITHM_LABELS)[self.algorithm_index]
        self.settings["algorithm"] = key_name
        self.algorithm_button.text = ALGORITHM_LABELS[key_name]

    def _set_uav_count(self, value: float) -> None:
        self.settings["n_uavs"] = int(value)
        self._invalidate_result()
        self._update_dynamic(force=True)

    def _set_sensing(self, value: float) -> None:
        self.settings["sensing_radius"] = float(value)
        self._invalidate_result()
        self._update_dynamic(force=True)

    def _set_communication(self, value: float) -> None:
        self.settings["communication_radius"] = float(value)
        self._invalidate_result()
        self._update_dynamic(force=True)

    def _toggle_sensing(self) -> None:
        self.show_sensing = not self.show_sensing
        self.sensing_button.text = f"Cảm biến: {'Bật' if self.show_sensing else 'Tắt'}"
        self._update_dynamic(force=True)

    def _toggle_links(self) -> None:
        self.show_links = not self.show_links
        self.links_button.text = f"Liên kết: {'Bật' if self.show_links else 'Tắt'}"
        self._update_dynamic(force=True)

    def _toggle_trails(self) -> None:
        self.show_trails = not self.show_trails
        self.trails_button.text = f"Vệt bay: {'Bật' if self.show_trails else 'Tắt'}"
        self.last_trail_index = -1
        self._update_dynamic(force=True)

    def _cycle_speed(self) -> None:
        speeds = [0.5, 1.0, 2.0, 5.0]
        self.playback_speed = speeds[(speeds.index(self.playback_speed) + 1) % len(speeds)]
        shown = int(self.playback_speed) if self.playback_speed.is_integer() else self.playback_speed
        self.speed_button.text = f"Tốc độ: {shown}x"

    def _generate_map(self) -> None:
        difficulty = ["Dễ", "Trung bình", "Khó"][self.difficulty_index]
        self.map_seed += 1
        self.city_map = generate_city_map(
            n_targets=self.target_count,
            n_obstacles=self.obstacle_count,
            difficulty=difficulty,
            seed=self.map_seed,
        )
        self._invalidate_result()
        self._rebuild_static()
        self._update_dynamic(force=True)
        self._set_status(f"Đã sinh map {difficulty.lower()} — seed {self.map_seed}")

    def _save_map(self) -> None:
        DEFAULT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_city_map(self.city_map, DEFAULT_MAP_PATH)
        self._set_status(f"Đã lưu {DEFAULT_MAP_PATH}")

    def _load_map(self) -> None:
        try:
            self.city_map = load_city_map(DEFAULT_MAP_PATH)
        except Exception as error:
            self._set_status(f"Không mở được map: {error}", error=True)
            return
        self.target_count = len(self.city_map.targets)
        self.obstacle_count = len(self.city_map.obstacles)
        self.target_slider.value = self.target_count
        self.obstacle_slider.value = self.obstacle_count
        self._invalidate_result()
        self._rebuild_static()
        self._update_dynamic(force=True)
        self._set_status(f"Đã mở {DEFAULT_MAP_PATH}")

    def _start_solver(self) -> None:
        if self.future is not None and not self.future.done():
            self._set_status("Một thuật toán đang chạy", error=True)
            return
        try:
            self.city_map.to_problem(**{
                name: self.settings[name]
                for name in (
                    "n_uavs", "sensing_radius", "communication_radius",
                    "min_separation", "max_speed", "obstacle_clearance",
                )
            })
        except Exception as error:
            self._set_status(f"Cấu hình không hợp lệ: {error}", error=True)
            return
        self.playing = False
        self.run_button.set_enabled(False)
        self.run_button.text = "ĐANG TỐI ƯU..."
        self._set_status("Thuật toán đang chạy ở tiến trình nền...")
        self.future = self.executor.submit(
            execute_simulation, self.city_map.to_dict(), dict(self.settings)
        )

    def _poll_solver(self) -> None:
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        self.run_button.set_enabled(True)
        self.run_button.text = "CHẠY TỐI ƯU"
        try:
            self.result = future.result()
        except Exception as error:
            self._set_status(f"Thuật toán lỗi: {error}", error=True)
            return
        trajectory = self.result["trajectory"]
        duration = max((len(trajectory) - 1) * self.result["dt"], 0.01)
        self.timeline.minimum = 0.0
        self.timeline.maximum = duration
        self.timeline.value = 0.0
        self.simulation_time = 0.0
        self.last_trail_index = -1
        self.playing = True
        self.play_button.text = "Dừng"
        self._update_metrics()
        self._update_dynamic(force=True)
        if self.result["transition_available"]:
            self._set_status(
                f"Hoàn tất {self.result['label']} trong {self.result['runtime']:.2f}s"
            )
        else:
            self._set_status("Không có đường đã kiểm chứng; UAV giữ tại đội hình đầu", error=True)

    def _update_metrics(self) -> None:
        if self.result is None:
            self.metrics_label.text = "Chưa có kết quả"
            return
        metrics = self.result["metrics"]
        self.metrics_label.text = (
            f"{self.result['label']}\n"
            f"Bao phủ: {metrics['weighted_coverage_ratio']:.1%}\n"
            f"Dư thừa: {metrics['redundancy_excess']:.3f}\n"
            f"Điểm chung: {metrics['joint_fitness']:.3f}\n"
            f"Thời gian đội hình: {metrics['formation_time_sec']:.1f} s\n"
            f"Quãng đường: {metrics['total_travel_distance']:.1f} m\n"
            f"Hợp lệ: {'Có' if metrics['feasible'] else 'Không'}"
        )

    def _toggle_play(self) -> None:
        if self.result is None:
            return
        self.playing = not self.playing
        self.play_button.text = "Dừng" if self.playing else "Phát"

    def _rewind(self) -> None:
        self.playing = False
        self.simulation_time = 0.0
        self.timeline.value = 0.0
        self.play_button.text = "Phát"
        self.last_trail_index = -1
        self._update_dynamic(force=True)

    def _seek(self, value: float) -> None:
        self.playing = False
        self.play_button.text = "Phát"
        self.simulation_time = float(value)
        self._update_dynamic(force=True)

    def _invalidate_result(self) -> None:
        self.result = None
        self.playing = False
        self.simulation_time = 0.0
        self.timeline.minimum = 0.0
        self.timeline.maximum = 1.0
        self.timeline.value = 0.0
        self.play_button.text = "Phát"
        self.last_trail_index = -1
        self._clear_trails()
        self._update_metrics()

    def _map_transform(self) -> tuple[float, float, float]:
        x, y, width, height = self.map_rect
        scale = min(
            (width - 2 * MAP_PADDING) / self.city_map.width,
            (height - 2 * MAP_PADDING) / self.city_map.height,
        )
        origin_x = x + (width - self.city_map.width * scale) / 2
        origin_y = y + (height - self.city_map.height * scale) / 2
        return origin_x, origin_y, scale

    def _to_screen(self, point) -> tuple[float, float]:
        ox, oy, scale = self._map_transform()
        return ox + float(point[0]) * scale, oy + float(point[1]) * scale

    def _from_screen(self, x: float, y: float) -> tuple[float, float] | None:
        ox, oy, scale = self._map_transform()
        map_x, map_y = (x - ox) / scale, (y - oy) / scale
        if 0 <= map_x <= self.city_map.width and 0 <= map_y <= self.city_map.height:
            return float(map_x), float(map_y)
        return None

    def _rebuild_static(self) -> None:
        self.static_batch = pyglet.graphics.Batch()
        self.target_shapes = []
        self.last_covered = None
        x, y, width, height = self.map_rect
        shapes.BorderedRectangle(
            x, y, width, height, border=2,
            color=MAP_BG, border_color=(83, 105, 133), batch=self.static_batch,
        )
        ox, oy, scale = self._map_transform()
        map_width = self.city_map.width * scale
        map_height = self.city_map.height * scale
        for fraction in np.linspace(0.1, 0.9, 9):
            shapes.Line(
                ox + fraction * map_width, oy,
                ox + fraction * map_width, oy + map_height,
                color=GRID, batch=self.static_batch,
            )
            shapes.Line(
                ox, oy + fraction * map_height,
                ox + map_width, oy + fraction * map_height,
                color=GRID, batch=self.static_batch,
            )
        for obstacle in self.city_map.obstacles:
            bottom_left = self._to_screen((obstacle.x_min, obstacle.y_min))
            rectangle = shapes.BorderedRectangle(
                bottom_left[0], bottom_left[1],
                (obstacle.x_max - obstacle.x_min) * scale,
                (obstacle.y_max - obstacle.y_min) * scale,
                border=2, color=(132, 57, 52), border_color=RED,
                batch=self.static_batch,
            )
            rectangle.opacity = 220
        max_weight = max(float(np.max(self.city_map.target_weights)), 1.0)
        for target, weight in zip(self.city_map.targets, self.city_map.target_weights):
            px, py = self._to_screen(target)
            target_shape = shapes.Circle(
                px, py, 3 + 3 * float(weight) / max_weight,
                segments=12, color=(148, 157, 173), batch=self.static_batch,
            )
            target_shape.opacity = 220
            self.target_shapes.append(target_shape)

    def _rebuild_dynamic(self, n_uavs: int) -> None:
        self.dynamic_batch = pyglet.graphics.Batch()
        self.sensing_shapes = []
        self.link_shapes = {}
        self.uav_shapes = []
        self.uav_labels = []
        for _ in range(n_uavs):
            circle = shapes.Circle(
                0, 0, 1, segments=48, color=(52, 128, 190), batch=self.dynamic_batch,
            )
            circle.opacity = 35
            self.sensing_shapes.append(circle)
        for start in range(n_uavs):
            for end in range(start + 1, n_uavs):
                line = shapes.Line(0, 0, 0, 0, thickness=2, color=BLUE, batch=self.dynamic_batch)
                line.opacity = 0
                self.link_shapes[(start, end)] = line
        for index in range(n_uavs):
            self.uav_shapes.append(
                shapes.Circle(0, 0, 9, segments=20, color=AMBER, batch=self.dynamic_batch)
            )
            self.uav_labels.append(
                pyglet.text.Label(
                    str(index), x=0, y=0, font_name="DejaVu Sans", font_size=9,
                    color=TEXT, batch=self.dynamic_batch,
                )
            )
        self.dynamic_uav_count = n_uavs

    def _current_positions(self) -> tuple[np.ndarray, int]:
        if self.result is None:
            problem = self.city_map.to_problem(**{
                name: self.settings[name]
                for name in (
                    "n_uavs", "sensing_radius", "communication_radius",
                    "min_separation", "max_speed", "obstacle_clearance",
                )
            })
            return problem.start_positions, 0
        trajectory = self.result["trajectory"]
        fractional = self.simulation_time / max(self.result["dt"], 1e-12)
        start = min(int(fractional), len(trajectory) - 1)
        end = min(start + 1, len(trajectory) - 1)
        alpha = min(max(fractional - start, 0.0), 1.0)
        return (1 - alpha) * trajectory[start] + alpha * trajectory[end], start

    def _clear_trails(self) -> None:
        self.trail_batch = pyglet.graphics.Batch()
        self.trail_shapes = []

    def _rebuild_trails(self, trajectory_index: int) -> None:
        self._clear_trails()
        if self.result is None or not self.show_trails:
            return
        trajectory = self.result["trajectory"]
        end = min(trajectory_index + 2, len(trajectory))
        if end < 2:
            return
        sample_count = min(end, 180)
        indices = np.unique(np.linspace(0, end - 1, sample_count, dtype=int))
        for uav_index in range(trajectory.shape[1]):
            points = [self._to_screen(trajectory[i, uav_index]) for i in indices]
            for first, second in zip(points, points[1:]):
                line = shapes.Line(
                    first[0], first[1], second[0], second[1],
                    thickness=2, color=AMBER, batch=self.trail_batch,
                )
                line.opacity = 150
                self.trail_shapes.append(line)

    def _update_dynamic(self, *, force: bool = False) -> None:
        try:
            positions, trajectory_index = self._current_positions()
        except Exception as error:
            self._set_status(f"Không thể vẽ: {error}", error=True)
            return
        if self.dynamic_uav_count != len(positions):
            self._rebuild_dynamic(len(positions))
            force = True
        sensing_radius = (
            self.result["sensing_radius"] if self.result else self.settings["sensing_radius"]
        )
        communication_radius = (
            self.result["communication_radius"]
            if self.result else self.settings["communication_radius"]
        )
        _, _, scale = self._map_transform()
        delta = self.city_map.targets[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        covered = np.any(distances <= sensing_radius, axis=1)
        if self.last_covered is None or not np.array_equal(covered, self.last_covered):
            for index, is_covered in enumerate(covered):
                self.target_shapes[index].color = GREEN[:3] if is_covered else (148, 157, 173)
            self.last_covered = covered.copy()

        graph = communication_graph_from_positions(positions, communication_radius)
        active_edges = {tuple(sorted(edge)) for edge in graph.edges()}
        for index, position in enumerate(positions):
            px, py = self._to_screen(position)
            sensing = self.sensing_shapes[index]
            sensing.position = px, py
            sensing.radius = sensing_radius * scale
            sensing.opacity = 35 if self.show_sensing else 0
            self.uav_shapes[index].position = px, py
            self.uav_labels[index].position = px + 11, py - 5, 0
        for edge, line in self.link_shapes.items():
            if self.show_links and edge in active_edges:
                first, second = self._to_screen(positions[edge[0]]), self._to_screen(positions[edge[1]])
                line.position = first
                line.x2, line.y2 = second
                line.opacity = 190
            else:
                line.opacity = 0
        if force or trajectory_index != self.last_trail_index:
            self._rebuild_trails(trajectory_index)
            self.last_trail_index = trajectory_index

        coverage = float(
            np.sum(self.city_map.target_weights[covered]) / np.sum(self.city_map.target_weights)
        )
        connected = nx.is_connected(graph)
        self.status_label.text = (
            f"{self.status}    |    t={self.simulation_time:.1f}s    |    "
            f"coverage={coverage:.1%}    |    liên lạc={'có' if connected else 'không'}"
        )

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.status = message
        self.status_label.text = message
        self.status_label.color = RED if error else MUTED

    def _edit_map(self, x: float, y: float) -> None:
        point = self._from_screen(x, y)
        if point is None or self.edit_mode_index == 0:
            return
        mode = self.edit_modes[self.edit_mode_index]
        map_x, map_y = point
        if mode == "Thêm mục tiêu":
            self.city_map.targets = np.vstack([self.city_map.targets, [map_x, map_y]])
            self.city_map.target_weights = np.append(self.city_map.target_weights, 1.0)
        elif mode == "Thêm vật cản":
            half_w, half_h = 45.0, 35.0
            self.city_map.obstacles = (*self.city_map.obstacles, AxisAlignedRectangle(
                max(0.0, map_x - half_w), max(0.0, map_y - half_h),
                min(self.city_map.width, map_x + half_w),
                min(self.city_map.height, map_y + half_h),
                name=f"user_obstacle_{len(self.city_map.obstacles) + 1}",
            ))
        elif mode == "Xóa":
            for index, obstacle in enumerate(self.city_map.obstacles):
                if obstacle.x_min <= map_x <= obstacle.x_max and obstacle.y_min <= map_y <= obstacle.y_max:
                    self.city_map.obstacles = tuple(
                        item for item_index, item in enumerate(self.city_map.obstacles)
                        if item_index != index
                    )
                    break
            else:
                if len(self.city_map.targets) > 1:
                    distances = np.linalg.norm(self.city_map.targets - [map_x, map_y], axis=1)
                    nearest = int(np.argmin(distances))
                    if distances[nearest] <= 30.0:
                        self.city_map.targets = np.delete(self.city_map.targets, nearest, axis=0)
                        self.city_map.target_weights = np.delete(self.city_map.target_weights, nearest)
        self.target_count = len(self.city_map.targets)
        self.obstacle_count = len(self.city_map.obstacles)
        self.target_slider.value = self.target_count
        self.obstacle_slider.value = self.obstacle_count
        self._invalidate_result()
        self._rebuild_static()
        self._update_dynamic(force=True)

    def update(self, dt: float) -> None:
        self._poll_solver()
        if self.result is not None and self.playing:
            duration = (len(self.result["trajectory"]) - 1) * self.result["dt"]
            self.simulation_time = min(duration, self.simulation_time + dt * self.playback_speed)
            self.timeline.value = self.simulation_time
            if self.simulation_time >= duration:
                self.playing = False
                self.play_button.text = "Phát"
            self._update_dynamic()
        self.invalid = True

    def on_draw(self) -> None:
        started = time.perf_counter()
        self.clear()
        self.static_batch.draw()
        self.trail_batch.draw()
        self.dynamic_batch.draw()
        self.ui_batch.draw()
        self.draw_ms = (time.perf_counter() - started) * 1000.0
        self.frame_count += 1
        elapsed = time.perf_counter() - self.fps_started
        if elapsed >= 0.5:
            fps = self.frame_count / elapsed
            self.latest_fps = fps
            self.fps_label.text = (
                f"FPS: {fps:.0f} | draw {self.draw_ms:.1f}ms | present {self.present_ms:.1f}ms"
            )
            self.fps_label.color = GREEN if fps >= 58 else AMBER
            self.frame_count = 0
            self.fps_started = time.perf_counter()

    def flip(self) -> None:
        started = time.perf_counter()
        super().flip()
        measured = (time.perf_counter() - started) * 1000.0
        self.present_ms = measured if self.present_ms == 0 else 0.85 * self.present_ms + 0.15 * measured

    def on_mouse_press(self, x, y, button, modifiers) -> None:
        if button != mouse.LEFT:
            return
        for control in reversed(self.buttons):
            if control.hit(x, y):
                self.active_button = control
                control.set_pressed(True)
                return
        for slider in reversed(self.sliders):
            if slider.hit(x, y):
                self.active_slider = slider
                slider.dragging = True
                slider.set_from_x(x)
                return
        self._edit_map(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if self.active_slider is not None and buttons & mouse.LEFT:
            self.active_slider.set_from_x(x)

    def on_mouse_release(self, x, y, button, modifiers) -> None:
        if self.active_button is not None:
            active = self.active_button
            self.active_button = None
            should_press = active.hit(x, y)
            active.set_pressed(False)
            if should_press:
                active.press()
        if self.active_slider is not None:
            self.active_slider.commit()
            self.active_slider.dragging = False
            self.active_slider = None

    def on_mouse_motion(self, x, y, dx, dy) -> None:
        for control in self.buttons:
            control.set_hovered(control.hit(x, y))

    def on_mouse_leave(self, x, y) -> None:
        if self.active_button is not None:
            self.active_button.set_pressed(False)
            self.active_button = None
        for control in self.buttons:
            control.set_hovered(False)

    def on_key_press(self, symbol, modifiers) -> None:
        if symbol == key.SPACE:
            self._toggle_play()
        elif symbol == key.R:
            self._start_solver()
        elif symbol == key.G:
            self._generate_map()
        elif symbol == key.HOME:
            self._rewind()

    def on_close(self) -> None:
        pyglet.clock.unschedule(self.update)
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().on_close()


def run_pyglet_simulator(
    *,
    benchmark_seconds: float | None = None,
    double_buffer: bool = True,
) -> None:
    window = PygletSwarmSimulator(
        double_buffer=double_buffer,
    )
    if benchmark_seconds is not None:
        def finish_benchmark(_dt: float) -> None:
            print(
                f"PYGLET_BENCHMARK size={window.width}x{window.height} "
                f"fps={window.latest_fps:.1f} draw_ms={window.draw_ms:.2f} "
                f"present_ms={window.present_ms:.2f}",
                flush=True,
            )
            window.close()

        pyglet.clock.schedule_once(finish_benchmark, benchmark_seconds)
    pyglet.app.run(interval=1 / 60.0)
