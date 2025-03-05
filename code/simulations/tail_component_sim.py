from manim import *
import numpy as np

# ------------------ Global Style Constants ------------------
FISH_COLOR = "#4169E1"      # Royal Blue
OUTLINE_COLOR = WHITE
OUTLINE_WIDTH = 7
# Create 6 colors (one per tail component) using a gradient.
TIME_SERIES_COLORS = color_gradient([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE], 6)

# ------------------ Configuration Variables ------------------
SAMPLE_FRACTION = 0.05      # Use this fraction of the full tail-data frames (e.g. 0.01 = 1%)
MAX_FRAMES = 10000             # Maximum number of frames to simulate (to keep the clip short)

# ------------------ Scene Definition ------------------
class FishTailAnimation(Scene):
    def construct(self):
        # ----- Zoom Out the Camera -----
        if hasattr(self.camera, "frame"):
            self.camera.frame.scale(8.0)
        else:
            print("Camera zoom adjustment not available; using config.frame_width and config.frame_height for zooming out.")

        # ----- Load and Subsample Tail Data -----
        # Each row is a time step; each column is a tail component angle (in radians).
        tail_data_full = np.load("tail_component_array.npy", allow_pickle=True)
        print("Full tail data shape:", tail_data_full.shape)
        skip = int(1 / SAMPLE_FRACTION)
        tail_data = tail_data_full[::skip]
        if tail_data.shape[0] > MAX_FRAMES:
            tail_data = tail_data[:MAX_FRAMES]
        total_frames = tail_data.shape[0]
        print("Subsampled tail data shape:", tail_data.shape)

        # ----- Incorporate Hand-Drawn Body Coordinates -----
        # These coordinates come directly from your HTML <area> tag in pixel space.
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

        # --- Normalization ---
        pixel_x_min, pixel_x_max = hand_drawn_coords[:, 0].min(), hand_drawn_coords[:, 0].max()
        pixel_y_min, pixel_y_max = hand_drawn_coords[:, 1].min(), hand_drawn_coords[:, 1].max()

        # Define Manim coordinate bounds.
        manim_x_min, manim_x_max = -3.5, 3.5
        manim_y_min, manim_y_max = -2, 2

        def normalize_coords(coords):
            x_norm = manim_x_min + (coords[:, 0] - pixel_x_min) / (pixel_x_max - pixel_x_min) * (manim_x_max - manim_x_min)
            y_norm = manim_y_min + (coords[:, 1] - pixel_y_min) / (pixel_y_max - pixel_y_min) * (manim_y_max - manim_y_min)
            return np.column_stack((x_norm, y_norm, np.zeros_like(x_norm)))

        normalized_coords = normalize_coords(hand_drawn_coords)

        # ----- Set Tail Attachment -----
        ta_pixel = np.array([1093, 325])
        ta_norm = normalize_coords(np.array([ta_pixel]))[0]
        base_body_coords = [list(coord - ta_norm) for coord in normalized_coords]
        if base_body_coords[0] != base_body_coords[-1]:
            base_body_coords.append(base_body_coords[0])
        base_body_rotation = 0 * DEGREES

        # ----- Compute the Fish Head -----
        self.fish_head = min(base_body_coords, key=lambda p: p[0])

        # ----- Create a ValueTracker for the Current Frame -----
        frame_tracker = ValueTracker(0)

        # ----- Create the Fish Components (tail, tail points, body) as always_redraw objects -----
        fish_group = VGroup(
            always_redraw(lambda: self.get_tail_polygon(tail_data, int(frame_tracker.get_value()))),
            always_redraw(lambda: self.get_tail_points(tail_data, int(frame_tracker.get_value()))),
            always_redraw(lambda: self.get_fish_body(tail_data, int(frame_tracker.get_value()),
                                                      base_body_coords, base_body_rotation))
        )

        # ----- Create the Time Series Panels as an always_redraw group -----
        # Note: Only the first (theta1) panel gets an x-axis label, placed above its axis.
        panels = always_redraw(lambda: VGroup(*[
            self.get_time_series_panel(
                tail_data,
                comp_index=i,
                current_frame=frame_tracker.get_value(),
                total_frames=total_frames,
                show_x_label=(i == 0)  # only panel for component 1 gets the label
            ) for i in range(6)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(RIGHT * 10))

        # ----- Group the fish and panels and move the whole group left
        main_group = VGroup(fish_group, panels)
        main_group.shift(LEFT * 3)
        self.add(main_group)

        # ----- On-Screen Text for Frame and Bout Count (with larger fonts) -----
        frame_text = Text("Frame: 0", font_size=36).to_corner(UL)
        bout_text = Text("Bouts: 0", font_size=36).next_to(frame_text, DOWN)
        self.add(frame_text, bout_text)

        # ----- Bout Detection (More Sensitive) -----
        self.bout_count = 0
        self.in_bout = False
        self.last_frame = 0
        bout_threshold = 0.1  # Reduced threshold

        def update_frame_text(mob):
            current_frame = int(frame_tracker.get_value())
            if current_frame > self.last_frame and current_frame > 0:
                diff = abs(tail_data[current_frame, 0] - tail_data[current_frame - 1, 0])
                if diff > bout_threshold and not self.in_bout:
                    self.bout_count += 1
                    self.in_bout = True
                elif diff <= bout_threshold:
                    self.in_bout = False
                self.last_frame = current_frame
            mob.become(Text(f"Frame: {current_frame}", font_size=36).to_corner(UL))
            return mob

        def update_bout_text(mob):
            mob.become(Text(f"Bouts: {self.bout_count}", font_size=36).next_to(frame_text, DOWN))
            return mob

        frame_text.add_updater(update_frame_text)
        bout_text.add_updater(update_bout_text)

        # ----- Animate the Clip -----
        self.play(
            frame_tracker.animate.increment_value(total_frames - 1),
            run_time=(total_frames - 1) / 60,
            rate_func=linear
        )
        self.wait()

    # ------------------ Helper Methods ------------------
    def get_tail_polygon(self, tail_data, frame):
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
            fill_color=FISH_COLOR,
            fill_opacity=1,
            stroke_color=OUTLINE_COLOR,
            stroke_width=OUTLINE_WIDTH
        )
        return tail_polygon

    def get_tail_points(self, tail_data, frame):
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
                point_color = FISH_COLOR
            else:
                point_color = TIME_SERIES_COLORS[i-1] if i-1 < len(TIME_SERIES_COLORS) else FISH_COLOR
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

    def get_fish_body(self, tail_data, frame, base_coords, base_rot):
        additional_rotation = tail_data[frame, 0]
        total_rotation = base_rot + additional_rotation
        poly = Polygon(
            *base_coords,
            fill_color=FISH_COLOR,
            fill_opacity=0.8,
            stroke_color=OUTLINE_COLOR,
            stroke_width=OUTLINE_WIDTH
        )
        poly.rotate(total_rotation, about_point=ORIGIN)
        return poly

    def get_time_series_panel(self, tail_data, comp_index, current_frame, total_frames,
                              panel_width=4, panel_height=2, show_x_label=False):
        """
        Creates a time series panel for tail component angle θ₍comp_index+1₎.
        Displays a sliding window (50 time steps) of the tail angle (y-axis: angle in radians,
        x-axis: time steps). A vertical yellow line indicates the current frame within that window.
        If show_x_label is True, an x-axis label ("Time step") is added above the axis.
        Additionally, a y-axis label (θ value) is added to the left.
        """
        window_length = 50
        current_frame_int = int(current_frame)
        t0 = max(0, current_frame_int - window_length)
        x_range = [0, window_length, window_length / 5]

        comp_data = tail_data[:, comp_index]
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

        # Plot the data for the window.
        x_vals = np.arange(t0, current_frame_int + 1)
        if len(x_vals) > 0:
            shifted_x_vals = x_vals - t0
            y_vals = tail_data[t0:current_frame_int+1, comp_index]
            points = [axes.c2p(x, y) for x, y in zip(shifted_x_vals, y_vals)]
            graph_line = VMobject()
            if points:
                graph_line.set_points_as_corners(points)
            graph_line.set_color(TIME_SERIES_COLORS[comp_index])
            graph_line.set_stroke(width=3)
        else:
            graph_line = VMobject()

        current_x = current_frame_int - t0
        v_line = axes.get_vertical_line(axes.c2p(current_x, y_min), color=YELLOW)
        v_line.set_stroke(width=2)

        # Add a y-axis label for the tail component.
        y_label = MathTex(f"\\theta_{{{comp_index+1}}}").scale(1.2)
        # Rotate counterclockwise 90°:
        y_label.rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2)

        panel_items = [axes, graph_line, v_line, y_label]

        if show_x_label:
            # Place the "Time step" label above the x-axis.
            x_label = Text("Time step", font_size=36)
            x_label.next_to(axes.x_axis, UP, buff=0.2)
            panel_items.append(x_label)

        panel = VGroup(*panel_items)
        return panel

    def rotate_point(self, point, angle):
        x, y, z = point
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z])

# ------------------ Main Block ------------------
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
