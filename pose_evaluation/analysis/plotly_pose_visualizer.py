import numpy as np
import plotly.graph_objects as go
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic


class PlotlyVisualizer(PoseVisualizer):
    def __init__(self, pose: Pose, thickness=None):
        super().__init__(pose=pose, thickness=thickness)
        self.background_color = np.array([255, 255, 255])

    def _get_plotly_point_trace(self, frame, confidence, component, idx_offset, min_confidence=0.0):
        xs, ys, zs, colors, texts = [], [], [], [], []

        for i, pt_name in enumerate(component.points):
            conf = confidence[i + idx_offset]
            if conf >= min_confidence:
                pt = frame[i + idx_offset]
                xs.append(pt[0])
                ys.append(pt[2])  # Z becomes Y
                zs.append(-pt[1])  # -Y becomes Z

                raw_color = np.array(component.colors[i % len(component.colors)])
                blended = raw_color * conf + self.background_color * (1 - conf)
                colors.append(f"rgba({int(blended[0])},{int(blended[1])},{int(blended[2])},{conf:.2f})")
                texts.append(
                    f"{component.name}:{pt_name}<br>X={pt[0]:.1f}<br>Y={pt[1]:.1f}<br>Z={pt[2] if len(pt) > 2 else 0:.1f}"
                )

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker={"size": 5 if self.thickness is None else self.thickness, "color": colors},
            text=texts,
            hoverinfo="text",
            name=component.name,
            showlegend=False,
        )

    def _get_plotly_limb_traces(self, frame, confidence, component, idx_offset, min_confidence=0.0):
        traces = []
        for p1, p2 in component.limbs:
            c1, c2 = confidence[p1 + idx_offset], confidence[p2 + idx_offset]
            if c1 >= min_confidence and c2 >= min_confidence:
                pt1 = frame[p1 + idx_offset]
                pt2 = frame[p2 + idx_offset]

                x_line = [pt1[0], pt2[0]]
                y_line = [pt1[2], pt2[2]]
                z_line = [-pt1[1], -pt2[1]]

                color = np.mean(
                    [
                        np.array(component.colors[p1 % len(component.colors)]),
                        np.array(component.colors[p2 % len(component.colors)]),
                    ],
                    axis=0,
                )
                color_str = f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

                traces.append(
                    go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode="lines",
                        line={"color": color_str, "width": 3 if self.thickness is None else self.thickness},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        return traces

    def _get_frame_traces(self, frame, confidence, min_confidence=0.0):
        traces = []
        idx = 0
        for component in self.pose.header.components:
            traces.append(self._get_plotly_point_trace(frame, confidence, component, idx, min_confidence))
            traces.extend(self._get_plotly_limb_traces(frame, confidence, component, idx, min_confidence))
            idx += len(component.points)
        return traces

    def get_3d_figure(self, frame_idx: int = 0, min_confidence: float = 0.0) -> go.Figure:
        frame = self.pose.body.data[frame_idx, 0]
        confidence = self.pose.body.confidence[frame_idx, 0]

        fig = go.Figure(data=self._get_frame_traces(frame, confidence, min_confidence))
        fig.update_layout(
            scene={
                # xaxis=dict(visible=False),
                "xaxis": {"title": "X"},
                # yaxis=dict(visible=False),
                "yaxis": {"title": "Y"},
                # zaxis=dict(visible=False),
                "zaxis": {"title": "Z"},
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    def get_3d_animation(self, min_confidence: float = 0.0) -> go.Figure:
        from plotly.graph_objs import Frame

        frames = []
        initial_data = []
        n_frames, n_people, n_points, _ = self.pose.body.data.shape

        for frame_idx in range(n_frames):
            frame = self.pose.body.data[frame_idx, 0]
            confidence = self.pose.body.confidence[frame_idx, 0]
            traces = self._get_frame_traces(frame, confidence, min_confidence)
            if frame_idx == 0:
                initial_data = traces
            frames.append(Frame(data=traces, name=str(frame_idx)))

        fig = go.Figure(
            data=initial_data,
            frames=frames,
            layout=go.Layout(
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 1000 / self.pose.body.fps}, "fromcurrent": True}],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}},
                                ],
                            },
                        ],
                    }
                ],
                sliders=[
                    {
                        "steps": [
                            {"method": "animate", "label": str(i), "args": [[str(i)], {"mode": "immediate"}]}
                            for i in range(n_frames)
                        ],
                        "transition": {"duration": 0},
                        "x": 0,
                        "y": -0.1,
                        "currentvalue": {"prefix": "Frame: "},
                    }
                ],
                scene={
                    # xaxis=dict(visible=False),
                    "xaxis": {"title": "X"},
                    # yaxis=dict(visible=False),
                    "yaxis": {"title": "Y"},
                    # zaxis=dict(visible=False),
                    "zaxis": {"title": "Z"},
                    "aspectmode": "auto",
                    "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.0}},
                },
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            ),
        )
        return fig


if __name__ == "__main__":
    with open("pose_evaluation/utils/test/test_data/mediapipe/standard_landmarks/colin-1-HOUSE.pose", "rb") as f:
        pose = Pose.read(f.read())
    print(pose)

    pose = reduce_holistic(pose)
    print(pose)

    viz = PlotlyVisualizer(pose)
    fig = viz.get_3d_figure(frame_idx=0)
    fig.show()

    fig_animation = viz.get_3d_animation()
    fig_animation.show()
    # fig.write_html("plotly_pose_visualized.html")
