import webbrowser

from manim import *
import numpy as np
import os

# changing models...
# MODEL_NAME = "GPT-2 Pretrained"
# MODEL_FILE = "final_predictions_pretrained.npy"

MODEL_NAME = "GPT-2 Blank"
MODEL_FILE = "final_predictions_scratch.npy"

# ------------------ Configuration Variables ------------------
SAMPLE_FRACTION = 1      # Fraction of full tail-data frames to use
MAX_FRAMES = 10000       # Maximum number of frames to simulate

# ------------------ Global Style Constants ------------------
# Ground truth fish colors (cool hues)
FISH_COLOR = "#4169E1"      # Royal Blue
# Prediction fish colors (warm hues, complementary to blue)
FISH_COLOR_PRED = ORANGE    # Using Manim's ORANGE (#FF862F)

OUTLINE_COLOR = WHITE
OUTLINE_WIDTH = 7
# Time-series colors for ground truth (using a cool-to-warm gradient)
TIME_SERIES_COLORS = color_gradient([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE], 6)
# For predictions, use a contrasting warm gradient.
TIME_SERIES_COLORS_PRED = color_gradient([ORANGE, GOLD, RED, MAROON, MAROON_D, MAROON_E], 6)

class FishTailAnimation(Scene):
    @property
    def construct(self):
        # ----- Zoom Out the Camera -----
        if hasattr(self.camera, "frame"):
            self.camera.frame.scale(8.0)
        else:
            print("Camera zoom adjustment not available.")

        # ----- Load and Subsample Data -----
        ground_truth = np.load("../predicting_with_llms/final_predictions_groundtruth_val.npy", allow_pickle=True)
        pred_vanil = np.load(f"../predicting_with_llms/{MODEL_FILE}", allow_pickle=True)
        print("Ground truth shape:", ground_truth.shape)
        print("Predicted shape:", pred_vanil.shape)
        skip = int(1 / SAMPLE_FRACTION)
        ground_truth = ground_truth[::skip]
        pred_vanil = pred_vanil[::skip]
        if ground_truth.shape[0] > MAX_FRAMES:
            ground_truth = ground_truth[:MAX_FRAMES]
            pred_vanil = pred_vanil[:MAX_FRAMES]
        total_frames = ground_truth.shape[0]
        print("Subsampled data shape:", ground_truth.shape)

        # ----- Compute Overall Pearson Correlation and Overall RMSE (optional) -----
        gt_flat = ground_truth.ravel()
        pred_flat = pred_vanil.ravel()
        pearson_r = np.corrcoef(gt_flat, pred_flat)[0, 1]
        total_rmse_val = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
        print(f"Overall Pearson r: {pearson_r:.3f}")
        print(f"Overall RMSE: {total_rmse_val:.3f}")

        # ----- Compute Per-Frame RMSE and Cumulative RMSE -----
        # error_data[t] = instantaneous RMSE at time step t, across all components
        error_data = np.sqrt(np.mean((ground_truth - pred_vanil) ** 2, axis=1))
        # cumulative_rmse[t] = sum of per-frame RMSE from frame 0 up to t
        cumulative_rmse = np.cumsum(error_data)

        # ----- Incorporate Hand-Drawn Body Coordinates -----
        hand_drawn_coords = np.array([
            (581,188), (567,168), (551,153), (531,142), (507,130), (487,119), (471,116),
            (451,116), (430,118), (409,121), (389,107), (370,90), (342,76), (312,74),
            (293,73), (267,73), (236,76), (210,84), (187,94), (164,106), (142,124),
            (129,141), (123,161), (124,174), (115,183), (106,198), (99,209), (93,221),
            (87,238), (81,255), (80,278), (80,296), (81,314), (81,330), (78,351),
            (78,365), (78,384), (82,398), (90,416), (96,434), (101,463), (110,481),
            (119,496), (135,512), (153,533), (169,538), (191,546), (207,553), (228,555),
            (248,558), (270,558), (289,561), (310,564), (331,558), (349,553), (361,543),
            (370,534), (384,523), (402,527), (415,527), (438,529), (454,531), (470,529),
            (486,523), (503,518), (525,512), (543,500), (555,486), (573,471), (587,479),
            (601,493), (614,505), (633,514), (653,525), (673,532), (693,545), (710,549),
            (732,555), (753,553), (777,553), (795,547), (818,542), (834,541), (849,539),
            (845,525), (815,522), (791,518), (757,516), (732,516), (710,511), (686,505),
            (670,499), (658,489), (649,476), (667,478), (682,484), (702,487), (727,493),
            (750,493), (771,496), (791,496), (818,490), (838,487), (855,481), (877,475),
            (899,465), (919,458), (938,449), (958,443), (975,435), (994,426), (1013,421),
            (1032,413), (1053,405), (1077,396), (1092,388), (1095,367), (1095,347),
            (1093,325), (1093,307), (1090,285), (1084,273), (1063,265), (1044,261),
            (1021,253), (1005,249), (984,246), (962,237), (938,227), (918,220), (898,213),
            (878,207), (857,203), (830,191), (809,181), (790,179), (769,173), (752,167),
            (731,164), (709,167), (687,173), (668,177), (656,179), (672,164), (692,155),
            (710,148), (727,139), (751,123), (776,117), (801,110), (814,103), (804,89),
            (778,88), (754,93), (729,96), (705,101), (687,113), (670,121), (651,128),
            (635,141), (620,155), (605,166), (595,184)
        ])
        pixel_x_min, pixel_x_max = hand_drawn_coords[:, 0].min(), hand_drawn_coords[:, 0].max()
        pixel_y_min, pixel_y_max = hand_drawn_coords[:, 1].min(), hand_drawn_coords[:, 1].max()
        # Manim coordinate bounds.
        manim_x_min, manim_x_max = -3.5, 3.5
        manim_y_min, manim_y_max = -2, 2

        def normalize_coords(coords):
            x_norm = manim_x_min + (coords[:, 0] - pixel_x_min) / (pixel_x_max - pixel_x_min) * (manim_x_max - manim_x_min)
            y_norm = manim_y_min + (coords[:, 1] - pixel_y_min) / (pixel_y_max - pixel_y_min) * (manim_y_max - manim_y_min)
            return np.column_stack((x_norm, y_norm, np.zeros_like(x_norm)))

        normalized_coords = normalize_coords(hand_drawn_coords)
        # Tail attachment point:
        ta_pixel = np.array([1093, 325])
        ta_norm = normalize_coords(np.array([ta_pixel]))[0]
        base_body_coords = [list(coord - ta_norm) for coord in normalized_coords]
        # Close the body polygon if needed
        if base_body_coords[0] != base_body_coords[-1]:
            base_body_coords.append(base_body_coords[0])
        base_body_rotation = 0 * DEGREES

        # Compute fish head (if needed)
        self.fish_head = min(base_body_coords, key=lambda p: p[0])

        # ----- Create a ValueTracker for the Current Frame -----
        frame_tracker = ValueTracker(0)

        # ----- Create the Two Fish as Containers with Updaters -----
        fish_true = VGroup()
        fish_true.add_updater(lambda mob: mob.become(
            VGroup(
                self.get_tail_polygon(ground_truth, int(frame_tracker.get_value())),
                self.get_tail_points(ground_truth, int(frame_tracker.get_value())),
                self.get_fish_body(ground_truth, int(frame_tracker.get_value()), base_body_coords, base_body_rotation)
            ).shift(LEFT * 12)
        ))

        fish_pred = VGroup()
        fish_pred.add_updater(lambda mob: mob.become(
            VGroup(
                self.get_tail_polygon(pred_vanil, int(frame_tracker.get_value()), is_pred=True),
                self.get_tail_points(pred_vanil, int(frame_tracker.get_value()), is_pred=True),
                self.get_fish_body(pred_vanil, int(frame_tracker.get_value()), base_body_coords, base_body_rotation, is_pred=True)
            ).shift(RIGHT * 12)
        ))

        # ----- Add Titles Above Each Fish -----
        fish_true_group = always_redraw(
            lambda: VGroup(
                Text("Ground Truth", font_size=36)
                .next_to(fish_true, UP, buff=0.5),
                fish_true
            )
        )

        fish_pred_group = always_redraw(
            lambda: VGroup(
                Text(f"Model Prediction {MODEL_NAME}", font_size=36)
                .next_to(fish_pred, UP, buff=0.5),
                fish_pred
            )
        )

        # ----- Create the Time-Series Panels (vertical stack of 6 subplots) -----
        panels = always_redraw(
            lambda: VGroup(*[
                self.get_time_series_panel(
                    ground_truth, pred_vanil, comp, frame_tracker.get_value(), total_frames,
                    show_x_label=(comp == 0)
                )
                for comp in range(6)
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        )

        # ----- Create the RMSE Error Panel separately -----
        error_panel = always_redraw(
            lambda: self.get_error_panel(
                ground_truth, pred_vanil, frame_tracker.get_value(), total_frames
            ).to_edge(UP, buff=0.25)
        )

        # ---------------------------------------------------------------
        #   Add a text box for overall correlation & RMSE in red (static)
        # ---------------------------------------------------------------
        metrics_text = Text(
            f"Pearson r: {pearson_r:.3f} | RMSE: {total_rmse_val:.3f}",
            font_size=36,
            color=RED
        )
        metrics_text.to_corner(DR)
        self.add(metrics_text)

        # ---------------------------------------------------------------
        #   Add a "Cumulative RMSE" text that updates each frame
        # ---------------------------------------------------------------
        cumulative_error_text = Text("Cumulative RMSE: 0.00", font_size=36).to_corner(UR)
        self.add(cumulative_error_text)

        # We update the text by looking up the cumsum value at the current frame.
        cumulative_error_text.add_updater(
            lambda m: m.become(
                Text(
                    f"Cumulative RMSE: {cumulative_rmse[int(frame_tracker.get_value())]:.2f}",
                    font_size=36,
                    color=WHITE
                ).to_corner(UR)
            )
        )

        # ---------------------------------------------------------------
        #     MANUAL POSITIONING TO AVOID OVERLAP
        # ---------------------------------------------------------------
        # We'll group fish and panels in a single row:
        fish_row = VGroup(fish_true_group, panels, fish_pred_group)
        fish_row.arrange(RIGHT, buff=10)

        # 2) Place the fish row below the RMSE panel by some vertical distance
        fish_row.next_to(error_panel, DOWN, buff=2)
        # Optionally, center the entire fish_row horizontally:
        fish_row.move_to(ORIGIN)

        # Now add both the RMSE panel and the row to the scene
        self.add(error_panel, fish_row)

        # Optional: On-screen frame counter.
        frame_text = Text("Frame: 0", font_size=36).to_corner(UL)
        self.add(frame_text)
        frame_text.add_updater(
            lambda mob: mob.become(
                Text(f"Frame: {int(frame_tracker.get_value())}", font_size=36).to_corner(UL)
            )
        )

        # ----- Animate the Clip at 30 fps -----
        self.play(
            frame_tracker.animate.increment_value(total_frames - 1),
            run_time=(total_frames - 1) / 30,
            rate_func=linear
        )
        self.wait()

    # ------------------ Helper Methods ------------------
    def get_tail_polygon(self, tail_data, frame, is_pred=False):
        angles = tail_data[frame]
        points = [np.array([0.0, 0.0, 0.0])]
        current_point = np.array([0.0, 0.0, 0.0])
        for angle in angles:
            step = np.array([np.cos(angle), np.sin(angle), 0.0])
            current_point = current_point + step
            points.append(current_point)
        pts = np.array(points)
        n_points = pts.shape[0]
        base_half_width = 0.5
        tip_half_width = 0.1
        fractions = np.linspace(0, 1, n_points)
        half_widths = base_half_width + (tip_half_width - base_half_width) * fractions
        # Compute tangents and normals
        tangents = []
        for i in range(n_points):
            if i == 0:
                tangent = pts[1] - pts[0]
            elif i == n_points - 1:
                tangent = pts[-1] - pts[-2]
            else:
                tangent = pts[i+1] - pts[i-1]
            tangent_2d = tangent[:2]
            norm = np.linalg.norm(tangent_2d)
            tangent_2d = tangent_2d / norm if norm != 0 else np.array([1, 0])
            tangents.append(tangent_2d)
        tangents = np.array(tangents)
        normals = np.array([[-t[1], t[0]] for t in tangents])
        left_points = pts.copy()
        right_points = pts.copy()
        for i in range(n_points):
            offset = normals[i] * half_widths[i]
            left_points[i][:2] = pts[i][:2] + offset
            right_points[i][:2] = pts[i][:2] - offset
        polygon_points = list(left_points) + list(right_points[::-1])
        tail_polygon = Polygon(
            *polygon_points,
            fill_color=FISH_COLOR_PRED if is_pred else FISH_COLOR,
            fill_opacity=1,
            stroke_color=OUTLINE_COLOR,
            stroke_width=OUTLINE_WIDTH
        )
        return tail_polygon

    def get_tail_points(self, tail_data, frame, is_pred=False):
        angles = tail_data[frame]
        points = [np.array([0.0, 0.0, 0.0])]
        current_point = np.array([0.0, 0.0, 0.0])
        for angle in angles:
            step = np.array([np.cos(angle), np.sin(angle), 0.0])
            current_point = current_point + step
            points.append(current_point)
        circles = VGroup()
        for i, pt in enumerate(points):
            if i == 0:
                point_color = FISH_COLOR_PRED if is_pred else FISH_COLOR
            else:
                color_idx = i - 1
                if is_pred:
                    if color_idx < len(TIME_SERIES_COLORS_PRED):
                        point_color = TIME_SERIES_COLORS_PRED[color_idx]
                    else:
                        point_color = FISH_COLOR_PRED
                else:
                    if color_idx < len(TIME_SERIES_COLORS):
                        point_color = TIME_SERIES_COLORS[color_idx]
                    else:
                        point_color = FISH_COLOR
            circ = Circle(
                radius=0.15,
                fill_color=point_color,
                fill_opacity=1,
                stroke_color=OUTLINE_COLOR,
                stroke_width=2
            )
            circ.move_to(pt)
            circles.add(circ)
        return circles

    def get_fish_body(self, tail_data, frame, base_coords, base_rot, is_pred=False):
        additional_rotation = tail_data[frame, 0]
        total_rotation = base_rot + additional_rotation
        poly = Polygon(
            *base_coords,
            fill_color=FISH_COLOR_PRED if is_pred else FISH_COLOR,
            fill_opacity=0.8,
            stroke_color=OUTLINE_COLOR,
            stroke_width=OUTLINE_WIDTH
        )
        poly.rotate(total_rotation, about_point=ORIGIN)
        return poly

    def get_time_series_panel(self, ground_truth, pred, comp_index, current_frame, total_frames,
                              panel_width=4, panel_height=2, show_x_label=False):
        window_length = 50
        current_frame_int = int(current_frame)
        t0 = max(0, current_frame_int - window_length)
        x_range = [0, window_length, window_length / 5]
        comp_data = ground_truth[:, comp_index]
        y_min = np.min(comp_data)
        y_max = np.max(comp_data)
        margin = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
        y_min -= margin
        y_max += margin

        axes = Axes(
            x_range=x_range,
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=panel_width,
            y_length=panel_height,
            tips=False,
            axis_config={"color": WHITE}
        )
        x_vals = np.arange(t0, current_frame_int + 1)
        shifted_x_vals = x_vals - t0

        # Ground truth curve (solid)
        y_vals_true = ground_truth[t0:current_frame_int+1, comp_index]
        points_true = [axes.c2p(x, y) for x, y in zip(shifted_x_vals, y_vals_true)]
        graph_line_true = VMobject()
        if points_true:
            graph_line_true.set_points_as_corners(points_true)
        graph_line_true.set_color(TIME_SERIES_COLORS[comp_index])
        graph_line_true.set_stroke(width=3)

        # Prediction curve (lighter) -- use the pred color scheme
        y_vals_pred = pred[t0:current_frame_int+1, comp_index]
        points_pred = [axes.c2p(x, y) for x, y in zip(shifted_x_vals, y_vals_pred)]
        graph_line_pred = VMobject()
        if points_pred:
            graph_line_pred.set_points_as_corners(points_pred)
        graph_line_pred.set_color(TIME_SERIES_COLORS_PRED[comp_index])
        graph_line_pred.set_stroke(width=3, opacity=0.5)

        # Vertical line
        current_x = current_frame_int - t0
        v_line = axes.get_vertical_line(axes.c2p(current_x, y_min), color=YELLOW)
        v_line.set_stroke(width=2)

        # Label on y-axis
        y_label = MathTex(f"\\theta_{{{comp_index+1}}}").scale(1.2)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2)

        panel_items = [axes, graph_line_true, graph_line_pred, v_line, y_label]

        # Optionally add an x-axis label
        if show_x_label:
            x_label = Text("Time step", font_size=36)
            x_label.next_to(axes.x_axis, UP, buff=0.2)
            panel_items.append(x_label)

        return VGroup(*panel_items)

    def get_error_panel(self, ground_truth, pred, current_frame, total_frames,
                        panel_width=6, panel_height=2, show_x_label=True):
        # Creates an error panel that plots a moving window (50 time steps) of the RMSE
        window_length = 50
        current_frame_int = int(current_frame)
        t0 = max(0, current_frame_int - window_length)
        x_range = [0, window_length, window_length / 5]

        error_data = np.sqrt(np.mean((ground_truth - pred)**2, axis=1))
        window_errors = error_data[t0:current_frame_int+1]
        y_min = np.min(window_errors)
        y_max = np.max(window_errors)
        margin = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
        y_min -= margin
        y_max += margin

        axes = Axes(
            x_range=x_range,
            y_range=[y_min, y_max, (y_max - y_min) / 5],
            x_length=panel_width,
            y_length=panel_height,
            tips=False,
            axis_config={"color": WHITE}
        )
        x_vals = np.arange(t0, current_frame_int+1)
        shifted_x_vals = x_vals - t0
        y_vals = error_data[t0:current_frame_int+1]
        points = [axes.c2p(x, y) for x, y in zip(shifted_x_vals, y_vals)]
        graph_line = VMobject()
        if points:
            graph_line.set_points_as_corners(points)
        graph_line.set_color(YELLOW)
        graph_line.set_stroke(width=3)

        current_x = current_frame_int - t0
        v_line = axes.get_vertical_line(axes.c2p(current_x, y_min), color=RED)
        v_line.set_stroke(width=2)

        y_label = Text("RMSE", font_size=24)
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2)

        panel_items = [axes, graph_line, v_line, y_label]

        if show_x_label:
            x_label = Text("Time step", font_size=24)
            x_label.next_to(axes.x_axis, UP, buff=0.2)
            panel_items.append(x_label)

        return VGroup(*panel_items)

    def rotate_point(self, point, angle):
        x, y, z = point
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z])

if __name__ == "__main__":
    from manim import config
    config.pixel_width = 1280
    config.pixel_height = 720
    config.frame_rate = 30
    config.media_dir = "./media"
    config.frame_width = 40
    config.frame_height = 22.5

    scene = FishTailAnimation()
    scene.render()
