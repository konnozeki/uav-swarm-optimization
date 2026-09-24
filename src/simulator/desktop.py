from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

import dearpygui.dearpygui as dpg
import networkx as nx
import numpy as np

from ..graph_ops import communication_graph_from_positions
from ..obstacles import AxisAlignedRectangle
from .city_map import CityMap, generate_city_map, load_city_map, save_city_map
from .engine import ALGORITHM_LABELS, execute_simulation

CANVAS_WIDTH = 850
CANVAS_HEIGHT = 650
CANVAS_PADDING = 18


class SwarmSimulatorApp:
    """GPU-rendered desktop UI around the existing optimization engine."""

    def __init__(self) -> None:
        self.city_map = generate_city_map()
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.future: Future | None = None
        self.result: dict | None = None
        self.playing = False
        self.simulation_time = 0.0
        self.last_frame_time = time.perf_counter()
        self.last_fps_update = self.last_frame_time
        self.fps_window_started = self.last_frame_time
        self.frames_in_window = 0
        self.dynamic_elapsed = 0.0
        self.last_update_ms = 0.0
        self.last_render_ms = 0.0
        self.dynamic_uav_count = 0
        self.dynamic_dirty = True
        self.last_covered: np.ndarray | None = None
        self.last_trail_index = -1
        self.history_rows = 0
        self._build_ui()
        self._redraw_static()
        self._redraw_dynamic()

    def _build_ui(self) -> None:
        self._load_font()
        self._build_theme()

        with dpg.file_dialog(
            show=False,
            callback=self._load_map_callback,
            tag="load_map_dialog",
            width=720,
            height=430,
        ):
            dpg.add_file_extension(".json", color=(84, 170, 255, 255))
        with dpg.file_dialog(
            show=False,
            callback=self._save_map_callback,
            tag="save_map_dialog",
            width=720,
            height=430,
        ):
            dpg.add_file_extension(".json", color=(84, 170, 255, 255))

        with dpg.window(tag="main_window", no_title_bar=True):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=285, height=-1, border=True):
                    self._build_controls()
                with dpg.child_window(width=875, height=-1, border=True):
                    dpg.add_text("BẢN ĐỒ MÔ PHỎNG", color=(107, 184, 255))
                    with dpg.drawlist(
                        width=CANVAS_WIDTH,
                        height=CANVAS_HEIGHT,
                        tag="map_canvas",
                    ):
                        with dpg.draw_layer(tag="static_layer"):
                            pass
                        with dpg.draw_layer(tag="dynamic_layer"):
                            pass
                    with dpg.item_handler_registry(tag="canvas_handlers"):
                        dpg.add_item_clicked_handler(
                            button=dpg.mvMouseButton_Left,
                            callback=self._canvas_clicked,
                        )
                    dpg.bind_item_handler_registry("map_canvas", "canvas_handlers")
                    self._build_timeline()
                with dpg.child_window(width=-1, height=-1, border=True):
                    self._build_results()

        dpg.set_primary_window("main_window", True)

    def _load_font(self) -> None:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        ]
        font_path = next((path for path in candidates if path.exists()), None)
        if font_path is None:
            return
        with dpg.font_registry():
            dpg.add_font(str(font_path), 16, tag="ui_font")
        dpg.bind_font("ui_font")

    def _build_theme(self) -> None:
        with dpg.theme(tag="app_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 23, 33))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (24, 31, 43))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (38, 91, 145))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (49, 119, 188))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (35, 45, 61))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (38, 91, 145))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 7, 6)
        dpg.bind_theme("app_theme")

    def _build_controls(self) -> None:
        dpg.add_text("UAV SWARM SIMULATOR", color=(107, 184, 255))
        dpg.add_text("Mô phỏng nhiệm vụ 2D bằng Python", color=(145, 155, 170))
        dpg.add_separator()
        dpg.add_text("Bản đồ thành phố")
        dpg.add_combo(
            ["Dễ", "Trung bình", "Khó"],
            default_value="Trung bình",
            label="Độ khó",
            tag="difficulty",
            width=145,
        )
        dpg.add_input_int(
            label="Mục tiêu",
            default_value=80,
            min_value=1,
            min_clamped=True,
            tag="target_count",
            width=145,
        )
        dpg.add_input_int(
            label="Vật cản",
            default_value=10,
            min_value=0,
            min_clamped=True,
            tag="obstacle_count",
            width=145,
        )
        dpg.add_input_int(
            label="Seed bản đồ", default_value=42, tag="map_seed", width=145
        )
        dpg.add_button(label="Sinh bản đồ", callback=self._generate_map, width=125)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Mở", callback=lambda: dpg.show_item("load_map_dialog")
            )
            dpg.add_button(
                label="Lưu", callback=lambda: dpg.show_item("save_map_dialog")
            )

        dpg.add_separator()
        dpg.add_text("Chỉnh sửa bằng chuột")
        dpg.add_combo(
            ["Chỉ xem", "Thêm mục tiêu", "Thêm vật cản", "Xóa"],
            default_value="Chỉ xem",
            tag="edit_mode",
            width=185,
        )
        dpg.add_input_float(
            label="Trọng số",
            default_value=1.0,
            min_value=0.1,
            min_clamped=True,
            tag="new_target_weight",
            width=125,
        )
        dpg.add_input_float(
            label="Rộng vật cản",
            default_value=90.0,
            min_value=5.0,
            min_clamped=True,
            tag="new_obstacle_width",
            width=125,
        )
        dpg.add_input_float(
            label="Cao vật cản",
            default_value=70.0,
            min_value=5.0,
            min_clamped=True,
            tag="new_obstacle_height",
            width=125,
        )

        dpg.add_separator()
        dpg.add_text("Cấu hình đàn UAV")
        dpg.add_slider_int(
            label="Số UAV",
            default_value=6,
            min_value=2,
            max_value=20,
            tag="n_uavs",
            width=155,
            callback=self._configuration_changed,
        )
        dpg.add_input_float(
            label="Bán kính cảm biến",
            default_value=175.0,
            min_value=1.0,
            min_clamped=True,
            tag="sensing_radius",
            width=135,
            callback=self._configuration_changed,
        )
        dpg.add_input_float(
            label="Bán kính liên lạc",
            default_value=280.0,
            min_value=1.0,
            min_clamped=True,
            tag="communication_radius",
            width=135,
            callback=self._configuration_changed,
        )
        dpg.add_input_float(
            label="Khoảng cách an toàn",
            default_value=3.0,
            min_value=0.0,
            min_clamped=True,
            tag="min_separation",
            width=135,
        )
        dpg.add_input_float(
            label="Tốc độ tối đa",
            default_value=40.0,
            min_value=0.1,
            min_clamped=True,
            tag="max_speed",
            width=135,
        )
        dpg.add_input_float(
            label="Lề vật cản",
            default_value=8.0,
            min_value=0.0,
            min_clamped=True,
            tag="obstacle_clearance",
            width=135,
        )

        dpg.add_separator()
        dpg.add_text("Thuật toán")
        self.algorithm_keys = list(ALGORITHM_LABELS)
        self.algorithm_by_label = {
            label: key for key, label in ALGORITHM_LABELS.items()
        }
        dpg.add_combo(
            list(self.algorithm_by_label),
            default_value=ALGORITHM_LABELS["transition-aware"],
            tag="algorithm_label",
            width=250,
        )
        dpg.add_input_int(
            label="Seed", default_value=0, tag="optimizer_seed", width=135
        )
        dpg.add_slider_int(
            label="Quần thể",
            default_value=18,
            min_value=4,
            max_value=80,
            tag="population",
            width=155,
        )
        dpg.add_slider_int(
            label="Số thế hệ",
            default_value=20,
            min_value=1,
            max_value=100,
            tag="generations",
            width=155,
        )
        dpg.add_button(
            label="CHẠY TỐI ƯU",
            callback=self._start_solver,
            width=250,
            height=36,
            tag="run_button",
        )
        dpg.add_loading_indicator(show=False, tag="solver_spinner", radius=2.5)
        dpg.add_text("Sẵn sàng", tag="status_text", wrap=250, color=(121, 214, 146))

    def _build_timeline(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Phát", callback=self._toggle_play, tag="play_button", width=65
            )
            dpg.add_button(label="Về đầu", callback=self._rewind, width=70)
            dpg.add_slider_float(
                default_value=0.0,
                min_value=0.0,
                max_value=1.0,
                width=540,
                tag="timeline",
                callback=self._timeline_changed,
                format="%.1f s",
            )
            dpg.add_combo(
                ["0.5x", "1x", "2x", "5x"],
                default_value="1x",
                tag="playback_speed",
                width=70,
            )
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="Vùng cảm biến",
                default_value=True,
                tag="show_sensing",
                callback=self._mark_dynamic_dirty,
            )
            dpg.add_checkbox(
                label="Liên kết",
                default_value=True,
                tag="show_links",
                callback=self._mark_dynamic_dirty,
            )
            dpg.add_checkbox(
                label="Vệt bay",
                default_value=True,
                tag="show_trails",
                callback=self._mark_dynamic_dirty,
            )
        with dpg.group(horizontal=True):
            dpg.add_combo(
                ["60", "90", "120", "Không giới hạn"],
                default_value="60",
                label="Cập nhật/giây",
                tag="target_fps",
                width=105,
            )
            dpg.add_text("FPS: —", tag="fps_text", color=(121, 214, 146))

    def _build_results(self) -> None:
        dpg.add_text("KẾT QUẢ", color=(107, 184, 255))
        dpg.add_separator()
        labels = [
            ("result_algorithm", "Thuật toán", "—"),
            ("result_coverage", "Độ bao phủ", "—"),
            ("result_redundancy", "Dư thừa", "—"),
            ("result_joint", "Điểm chung", "—"),
            ("result_time", "Thời gian đội hình", "—"),
            ("result_travel", "Quãng đường", "—"),
            ("result_connected", "Liên lạc hiện tại", "—"),
            ("result_separation", "Khoảng cách nhỏ nhất", "—"),
            ("result_feasible", "Nghiệm hợp lệ", "—"),
            ("result_runtime", "Thời gian tính", "—"),
        ]
        for tag, label, value in labels:
            dpg.add_text(label, color=(145, 155, 170))
            dpg.add_text(value, tag=tag, color=(235, 239, 247))

        dpg.add_separator()
        dpg.add_text("Lịch sử chạy")
        with dpg.table(
            tag="history_table",
            header_row=True,
            row_background=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            scrollY=True,
            height=245,
        ):
            dpg.add_table_column(
                label="Thuật toán", width_fixed=True, init_width_or_weight=125
            )
            dpg.add_table_column(label="Cov", width_fixed=True, init_width_or_weight=52)
            dpg.add_table_column(
                label="Joint", width_fixed=True, init_width_or_weight=52
            )
            dpg.add_table_column(label="CPU", width_fixed=True, init_width_or_weight=52)

        dpg.add_separator()
        dpg.add_text("Trạng thái khung hình", color=(145, 155, 170))
        dpg.add_text("—", tag="frame_status", wrap=240)

    def _settings(self) -> dict:
        label = dpg.get_value("algorithm_label")
        return {
            "algorithm": self.algorithm_by_label[label],
            "n_uavs": int(dpg.get_value("n_uavs")),
            "sensing_radius": float(dpg.get_value("sensing_radius")),
            "communication_radius": float(dpg.get_value("communication_radius")),
            "min_separation": float(dpg.get_value("min_separation")),
            "max_speed": float(dpg.get_value("max_speed")),
            "obstacle_clearance": float(dpg.get_value("obstacle_clearance")),
            "seed": int(dpg.get_value("optimizer_seed")),
            "population": int(dpg.get_value("population")),
            "generations": int(dpg.get_value("generations")),
        }

    def _generate_map(self) -> None:
        try:
            self.city_map = generate_city_map(
                n_targets=int(dpg.get_value("target_count")),
                n_obstacles=int(dpg.get_value("obstacle_count")),
                difficulty=dpg.get_value("difficulty"),
                seed=int(dpg.get_value("map_seed")),
            )
        except Exception as error:
            self._set_status(f"Không thể sinh bản đồ: {error}", error=True)
            return
        self._invalidate_result()
        self._redraw_static()
        self._redraw_dynamic()
        self._set_status(
            f"Đã sinh {len(self.city_map.targets)} mục tiêu, "
            f"{len(self.city_map.obstacles)} vật cản"
        )

    def _configuration_changed(self) -> None:
        if self.result is None:
            self.dynamic_dirty = True
            self._redraw_dynamic()

    def _mark_dynamic_dirty(self) -> None:
        self.dynamic_dirty = True

    def _load_map_callback(self, sender, app_data) -> None:
        try:
            self.city_map = load_city_map(app_data["file_path_name"])
        except Exception as error:
            self._set_status(f"Không mở được bản đồ: {error}", error=True)
            return
        dpg.set_value("target_count", len(self.city_map.targets))
        dpg.set_value("obstacle_count", len(self.city_map.obstacles))
        dpg.set_value("map_seed", self.city_map.seed)
        if self.city_map.difficulty in ("Dễ", "Trung bình", "Khó"):
            dpg.set_value("difficulty", self.city_map.difficulty)
        self._invalidate_result()
        self._redraw_static()
        self._redraw_dynamic()
        self._set_status(f"Đã mở {app_data['file_path_name']}")

    def _save_map_callback(self, sender, app_data) -> None:
        path = Path(app_data["file_path_name"])
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            save_city_map(self.city_map, path)
        except Exception as error:
            self._set_status(f"Không lưu được bản đồ: {error}", error=True)
            return
        self._set_status(f"Đã lưu {path}")

    def _canvas_clicked(self) -> None:
        mode = dpg.get_value("edit_mode")
        if mode == "Chỉ xem":
            return
        mouse_x, mouse_y = dpg.get_drawing_mouse_pos()
        point = self._from_screen((mouse_x, mouse_y))
        if point is None:
            return
        x, y = point

        if mode == "Thêm mục tiêu":
            self.city_map.targets = np.vstack([self.city_map.targets, [x, y]])
            self.city_map.target_weights = np.append(
                self.city_map.target_weights,
                float(dpg.get_value("new_target_weight")),
            )
        elif mode == "Thêm vật cản":
            width = float(dpg.get_value("new_obstacle_width"))
            height = float(dpg.get_value("new_obstacle_height"))
            rectangle = AxisAlignedRectangle(
                max(0.0, x - width / 2),
                max(0.0, y - height / 2),
                min(self.city_map.width, x + width / 2),
                min(self.city_map.height, y + height / 2),
                name=f"user_obstacle_{len(self.city_map.obstacles) + 1}",
            )
            self.city_map.obstacles = (*self.city_map.obstacles, rectangle)
        elif mode == "Xóa":
            for index, obstacle in enumerate(self.city_map.obstacles):
                if (
                    obstacle.x_min <= x <= obstacle.x_max
                    and obstacle.y_min <= y <= obstacle.y_max
                ):
                    self.city_map.obstacles = tuple(
                        item
                        for item_index, item in enumerate(self.city_map.obstacles)
                        if item_index != index
                    )
                    break
            else:
                if len(self.city_map.targets) > 1:
                    distances = np.linalg.norm(self.city_map.targets - [x, y], axis=1)
                    target_index = int(np.argmin(distances))
                    if distances[target_index] <= 30.0:
                        self.city_map.targets = np.delete(
                            self.city_map.targets, target_index, axis=0
                        )
                        self.city_map.target_weights = np.delete(
                            self.city_map.target_weights, target_index
                        )

        dpg.set_value("target_count", len(self.city_map.targets))
        dpg.set_value("obstacle_count", len(self.city_map.obstacles))
        self._invalidate_result()
        self._redraw_static()
        self._redraw_dynamic()

    def _start_solver(self) -> None:
        if self.future is not None and not self.future.done():
            self._set_status("Một thuật toán đang chạy", error=True)
            return
        try:
            settings = self._settings()
            self.city_map.to_problem(
                n_uavs=settings["n_uavs"],
                sensing_radius=settings["sensing_radius"],
                communication_radius=settings["communication_radius"],
                min_separation=settings["min_separation"],
                max_speed=settings["max_speed"],
                obstacle_clearance=settings["obstacle_clearance"],
            )
        except Exception as error:
            self._set_status(f"Cấu hình không hợp lệ: {error}", error=True)
            return

        self.playing = False
        dpg.configure_item("play_button", label="Phát")
        dpg.configure_item("solver_spinner", show=True)
        dpg.configure_item("run_button", enabled=False)
        self._set_status("Đang tối ưu trong tiến trình nền...")
        self.future = self.executor.submit(
            execute_simulation,
            self.city_map.to_dict(),
            settings,
        )

    def _poll_solver(self) -> None:
        if self.future is None or not self.future.done():
            return
        future = self.future
        self.future = None
        dpg.configure_item("solver_spinner", show=False)
        dpg.configure_item("run_button", enabled=True)
        try:
            self.result = future.result()
        except Exception as error:
            self._set_status(f"Thuật toán lỗi: {error}", error=True)
            return

        trajectory = self.result["trajectory"]
        duration = max(0.0, (len(trajectory) - 1) * self.result["dt"])
        self.simulation_time = 0.0
        dpg.configure_item("timeline", max_value=max(duration, 0.01))
        dpg.set_value("timeline", 0.0)
        self.playing = True
        self.dynamic_dirty = True
        self.last_trail_index = -1
        dpg.configure_item("play_button", label="Dừng")
        self._update_result_labels()
        self._append_history()
        if self.result["transition_available"]:
            self._set_status(
                f"Hoàn tất {self.result['label']} trong {self.result['runtime']:.2f}s"
            )
        else:
            self._set_status(
                "Không có đường đã kiểm chứng; UI giữ UAV tại đội hình ban đầu",
                error=True,
            )

    def _toggle_play(self) -> None:
        if self.result is None:
            return
        self.playing = not self.playing
        dpg.configure_item("play_button", label="Dừng" if self.playing else "Phát")
        self.dynamic_dirty = True

    def _rewind(self) -> None:
        self.simulation_time = 0.0
        dpg.set_value("timeline", 0.0)
        self.dynamic_dirty = True
        self._redraw_dynamic()

    def _timeline_changed(self, sender, value) -> None:
        self.simulation_time = float(value)
        self.playing = False
        dpg.configure_item("play_button", label="Phát")
        self.dynamic_dirty = True
        self._redraw_dynamic()

    def _invalidate_result(self) -> None:
        self.result = None
        self.playing = False
        self.simulation_time = 0.0
        dpg.set_value("timeline", 0.0)
        dpg.configure_item("timeline", max_value=1.0)
        dpg.configure_item("play_button", label="Phát")
        self.dynamic_dirty = True

    def _update_result_labels(self) -> None:
        metrics = self.result["metrics"]
        values = {
            "result_algorithm": self.result["label"],
            "result_coverage": f"{metrics['weighted_coverage_ratio']:.1%}",
            "result_redundancy": f"{metrics['redundancy_excess']:.3f}",
            "result_joint": f"{metrics['joint_fitness']:.3f}",
            "result_time": f"{metrics['formation_time_sec']:.1f} giây",
            "result_travel": f"{metrics['total_travel_distance']:.1f} m",
            "result_feasible": "Có" if metrics["feasible"] else "Không",
            "result_runtime": f"{self.result['runtime']:.2f} giây",
        }
        for tag, value in values.items():
            dpg.set_value(tag, value)

    def _append_history(self) -> None:
        metrics = self.result["metrics"]
        self.history_rows += 1
        with dpg.table_row(parent="history_table"):
            dpg.add_text(self.result["label"], wrap=120)
            dpg.add_text(f"{metrics['weighted_coverage_ratio']:.2f}")
            dpg.add_text(f"{metrics['joint_fitness']:.2f}")
            dpg.add_text(f"{self.result['runtime']:.2f}")

    def _playback_multiplier(self) -> float:
        return {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}[
            dpg.get_value("playback_speed")
        ]

    def _current_positions(self) -> tuple[np.ndarray, int]:
        if self.result is None:
            settings = self._settings()
            problem = self.city_map.to_problem(
                n_uavs=settings["n_uavs"],
                sensing_radius=settings["sensing_radius"],
                communication_radius=settings["communication_radius"],
                min_separation=settings["min_separation"],
                max_speed=settings["max_speed"],
                obstacle_clearance=settings["obstacle_clearance"],
            )
            return problem.start_positions, 0

        trajectory = self.result["trajectory"]
        dt = self.result["dt"]
        fractional_index = self.simulation_time / max(dt, 1e-12)
        start_index = min(int(fractional_index), len(trajectory) - 1)
        end_index = min(start_index + 1, len(trajectory) - 1)
        alpha = min(max(fractional_index - start_index, 0.0), 1.0)
        positions = (1.0 - alpha) * trajectory[start_index] + alpha * trajectory[
            end_index
        ]
        return positions, start_index

    def _map_transform(self) -> tuple[float, float, float]:
        scale = min(
            (CANVAS_WIDTH - 2 * CANVAS_PADDING) / self.city_map.width,
            (CANVAS_HEIGHT - 2 * CANVAS_PADDING) / self.city_map.height,
        )
        used_width = self.city_map.width * scale
        used_height = self.city_map.height * scale
        origin_x = (CANVAS_WIDTH - used_width) / 2
        origin_y = (CANVAS_HEIGHT - used_height) / 2
        return origin_x, origin_y, scale

    def _to_screen(self, point) -> tuple[float, float]:
        origin_x, origin_y, scale = self._map_transform()
        return (
            origin_x + float(point[0]) * scale,
            origin_y + (self.city_map.height - float(point[1])) * scale,
        )

    def _from_screen(self, point) -> tuple[float, float] | None:
        origin_x, origin_y, scale = self._map_transform()
        x = (point[0] - origin_x) / scale
        y = self.city_map.height - (point[1] - origin_y) / scale
        if not (0 <= x <= self.city_map.width and 0 <= y <= self.city_map.height):
            return None
        return float(x), float(y)

    def _redraw_static(self) -> None:
        dpg.delete_item("static_layer", children_only=True)
        self.last_covered = None
        self.dynamic_dirty = True
        origin_x, origin_y, scale = self._map_transform()
        map_min = self._to_screen((0, self.city_map.height))
        map_max = self._to_screen((self.city_map.width, 0))
        dpg.draw_rectangle(
            map_min,
            map_max,
            color=(83, 105, 133, 255),
            fill=(14, 19, 27, 255),
            thickness=2,
            parent="static_layer",
        )
        for fraction in np.linspace(0.1, 0.9, 9):
            x = origin_x + fraction * self.city_map.width * scale
            y = origin_y + fraction * self.city_map.height * scale
            dpg.draw_line(
                (x, map_min[1]),
                (x, map_max[1]),
                color=(43, 52, 66, 130),
                parent="static_layer",
            )
            dpg.draw_line(
                (map_min[0], y),
                (map_max[0], y),
                color=(43, 52, 66, 130),
                parent="static_layer",
            )

        for obstacle in self.city_map.obstacles:
            dpg.draw_rectangle(
                self._to_screen((obstacle.x_min, obstacle.y_max)),
                self._to_screen((obstacle.x_max, obstacle.y_min)),
                color=(235, 111, 92, 255),
                fill=(132, 57, 52, 210),
                thickness=1.5,
                parent="static_layer",
            )

        max_weight = max(float(np.max(self.city_map.target_weights)), 1.0)
        for index, (target, weight) in enumerate(
            zip(self.city_map.targets, self.city_map.target_weights)
        ):
            radius = 3.0 + 3.0 * float(weight) / max_weight
            dpg.draw_circle(
                self._to_screen(target),
                radius,
                color=(235, 235, 238, 230),
                fill=(148, 157, 173, 210),
                tag=f"map_target_{index}",
                parent="static_layer",
            )

    def _rebuild_dynamic_items(self, n_uavs: int) -> None:
        dpg.delete_item("dynamic_layer", children_only=True)
        for index in range(n_uavs):
            dpg.draw_circle(
                (0, 0),
                1,
                color=(71, 181, 255, 75),
                fill=(52, 128, 190, 22),
                thickness=1,
                show=False,
                tag=f"sensing_{index}",
                parent="dynamic_layer",
            )
            dpg.draw_polyline(
                [(0, 0), (0, 0)],
                color=(255, 205, 84, 150),
                thickness=2,
                show=False,
                tag=f"trail_{index}",
                parent="dynamic_layer",
            )
        for start in range(n_uavs):
            for end in range(start + 1, n_uavs):
                dpg.draw_line(
                    (0, 0),
                    (0, 0),
                    color=(68, 164, 255, 190),
                    thickness=2,
                    show=False,
                    tag=f"link_{start}_{end}",
                    parent="dynamic_layer",
                )
        for index in range(n_uavs):
            dpg.draw_circle(
                (0, 0),
                9,
                color=(255, 245, 190, 255),
                fill=(245, 185, 52, 255),
                thickness=2,
                tag=f"uav_{index}",
                parent="dynamic_layer",
            )
            dpg.draw_text(
                (0, 0),
                str(index),
                color=(255, 244, 205, 255),
                size=14,
                tag=f"uav_label_{index}",
                parent="dynamic_layer",
            )
        self.dynamic_uav_count = n_uavs
        self.last_trail_index = -1

    def _redraw_dynamic(self) -> None:
        force_update = self.dynamic_dirty
        try:
            positions, trajectory_index = self._current_positions()
        except Exception as error:
            self._set_status(f"Không thể vẽ cấu hình: {error}", error=True)
            return
        if self.dynamic_uav_count != len(positions):
            self._rebuild_dynamic_items(len(positions))
            force_update = True
        settings = self._settings()
        sensing_radius = (
            self.result["sensing_radius"] if self.result else settings["sensing_radius"]
        )
        communication_radius = (
            self.result["communication_radius"]
            if self.result
            else settings["communication_radius"]
        )
        _, _, scale = self._map_transform()

        delta = self.city_map.targets[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        covered = np.any(distances <= sensing_radius, axis=1)
        if self.last_covered is None or not np.array_equal(covered, self.last_covered):
            changed = (
                range(len(covered))
                if self.last_covered is None
                else np.flatnonzero(covered != self.last_covered)
            )
            for index in changed:
                is_covered = bool(covered[index])
                dpg.configure_item(
                    f"map_target_{index}",
                    color=(82, 224, 145, 255) if is_covered else (235, 235, 238, 230),
                    fill=(82, 224, 145, 210) if is_covered else (148, 157, 173, 210),
                )
            self.last_covered = covered.copy()

        show_sensing = dpg.get_value("show_sensing")
        for index, position in enumerate(positions):
            dpg.configure_item(
                f"sensing_{index}",
                center=self._to_screen(position),
                radius=sensing_radius * scale,
                show=show_sensing,
            )

        graph = communication_graph_from_positions(positions, communication_radius)
        active_edges = {tuple(sorted(edge)) for edge in graph.edges()}
        show_links = dpg.get_value("show_links")
        for start in range(len(positions)):
            for end in range(start + 1, len(positions)):
                visible = show_links and (start, end) in active_edges
                dpg.configure_item(
                    f"link_{start}_{end}",
                    p1=self._to_screen(positions[start]),
                    p2=self._to_screen(positions[end]),
                    show=visible,
                )

        show_trails = self.result is not None and dpg.get_value("show_trails")
        if show_trails:
            trajectory = self.result["trajectory"]
            trail_end = min(trajectory_index + 2, len(trajectory))
            for uav_index in range(trajectory.shape[1]):
                if force_update or trajectory_index != self.last_trail_index:
                    points = [
                        self._to_screen(point)
                        for point in trajectory[:trail_end, uav_index, :]
                    ]
                    if len(points) < 2:
                        points = [points[0], points[0]]
                    dpg.configure_item(
                        f"trail_{uav_index}",
                        points=points,
                    )
                dpg.configure_item(f"trail_{uav_index}", show=True)
            self.last_trail_index = trajectory_index
        else:
            for uav_index in range(len(positions)):
                dpg.configure_item(f"trail_{uav_index}", show=False)

        for index, position in enumerate(positions):
            screen = self._to_screen(position)
            dpg.configure_item(
                f"uav_{index}",
                center=screen,
            )
            dpg.configure_item(
                f"uav_label_{index}",
                pos=(screen[0] + 10, screen[1] - 15),
            )

        min_distance = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                min_distance = min(
                    min_distance,
                    float(np.linalg.norm(positions[i] - positions[j])),
                )
        coverage = float(
            np.sum(self.city_map.target_weights[covered])
            / np.sum(self.city_map.target_weights)
        )
        connected = nx.is_connected(graph)
        dpg.set_value("result_connected", "Có" if connected else "Không")
        dpg.set_value(
            "result_separation",
            "∞" if not np.isfinite(min_distance) else f"{min_distance:.1f} m",
        )
        dpg.set_value(
            "frame_status",
            f"t={self.simulation_time:.1f}s | coverage={coverage:.1%} | "
            f"connected={'có' if connected else 'không'}",
        )
        self.dynamic_dirty = False

    def _set_status(self, message: str, *, error: bool = False) -> None:
        dpg.set_value("status_text", message)
        dpg.configure_item(
            "status_text",
            color=(244, 118, 108) if error else (121, 214, 146),
        )

    def frame(self) -> None:
        update_started = time.perf_counter()
        now = time.perf_counter()
        elapsed = min(now - self.last_frame_time, 0.1)
        self.last_frame_time = now
        self.dynamic_elapsed += elapsed
        self._poll_solver()

        simulation_advanced = False
        if self.result is not None and self.playing:
            simulation_advanced = True
            duration = (len(self.result["trajectory"]) - 1) * self.result["dt"]
            self.simulation_time += elapsed * self._playback_multiplier()
            if self.simulation_time >= duration:
                self.simulation_time = duration
                self.playing = False
                dpg.configure_item("play_button", label="Phát")
            dpg.set_value("timeline", self.simulation_time)

        selected_rate = dpg.get_value("target_fps")
        update_interval = (
            0.0
            if selected_rate == "Không giới hạn"
            else 1.0 / max(int(selected_rate), 1)
        )
        should_update_motion = (
            simulation_advanced and self.dynamic_elapsed + 1e-12 >= update_interval
        )
        if should_update_motion or self.dynamic_dirty:
            self._redraw_dynamic()
            self.dynamic_elapsed = 0.0

        self.frames_in_window += 1
        if now - self.last_fps_update >= 0.5:
            window_duration = max(now - self.fps_window_started, 1e-9)
            measured = self.frames_in_window / window_duration
            dpg.set_value(
                "fps_text",
                f"FPS: {measured:.0f} | cập nhật {self.last_update_ms:.1f}ms | "
                f"render {self.last_render_ms:.1f}ms",
            )
            dpg.configure_item(
                "fps_text",
                color=(121, 214, 146) if measured >= 58 else (244, 180, 85),
            )
            self.last_fps_update = now
            self.fps_window_started = now
            self.frames_in_window = 0
        self.last_update_ms = (time.perf_counter() - update_started) * 1000.0

    def record_render_time(self, seconds: float) -> None:
        milliseconds = seconds * 1000.0
        if self.last_render_ms == 0.0:
            self.last_render_ms = milliseconds
        else:
            self.last_render_ms = 0.85 * self.last_render_ms + 0.15 * milliseconds

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def run_desktop_simulator() -> None:
    dpg.create_context()
    app = SwarmSimulatorApp()
    dpg.create_viewport(
        title="UAV Swarm Formation & Coverage Simulator",
        width=1500,
        height=880,
        min_width=1280,
        min_height=760,
        vsync=True,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_viewport_vsync(True)
    try:
        while dpg.is_dearpygui_running():
            app.frame()
            render_started = time.perf_counter()
            dpg.render_dearpygui_frame()
            app.record_render_time(time.perf_counter() - render_started)
    finally:
        app.close()
        dpg.destroy_context()
