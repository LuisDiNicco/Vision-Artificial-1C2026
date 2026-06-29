from .help import add_help_modal
from .recognition_panel import add_recognition_tab
from .source_panel import add_source_tab
from .training_panel import add_training_tab
from .video_panel import add_video_analysis_tab


def build_main_window(dpg, owner, video_w: int, video_h: int, video_texture_tag: str) -> None:
    with dpg.window(tag="main_window", label="TP Integrador - Reconocimiento Facial", no_close=True):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="sidebar_panel", width=340, height=820, border=True):
                dpg.add_text("TP Integrador", tag="title_text", color=(245, 247, 250))
                dpg.add_text("Reconocimiento facial", tag="subtitle_text", color=(120, 210, 255))
                dpg.add_separator()

                with dpg.tab_bar(tag="workflow_tabs", callback=owner._on_workflow_tab_changed):
                    add_training_tab(dpg, owner)
                    add_recognition_tab(dpg, owner)
                    add_video_analysis_tab(dpg, owner)
                    add_source_tab(dpg, owner)

                dpg.add_separator()
                dpg.add_text("Estado")
                dpg.add_text("", tag="status_text", wrap=300, color=(225, 230, 235))
                dpg.add_text("", tag="stats_text", wrap=300, color=(160, 170, 180))

            with dpg.child_window(tag="video_panel", width=1308, height=820, border=False):
                dpg.add_image(video_texture_tag, tag="video_image", width=video_w, height=video_h)
                with dpg.group(tag="video_playback_controls", show=False):
                    dpg.add_slider_float(
                        tag="video_seek_slider",
                        callback=owner._seek_actor_video,
                        min_value=0.0,
                        max_value=1.0,
                        default_value=0.0,
                        format="",
                        enabled=False,
                        width=max(video_w, 240),
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="<< 5s",
                            tag="video_skip_back_button",
                            callback=owner._skip_actor_video,
                            user_data=-5.0,
                            enabled=False,
                            width=78,
                        )
                        dpg.add_button(
                            label=">",
                            tag="video_play_pause_button",
                            callback=owner._toggle_actor_video_playback,
                            enabled=False,
                            width=54,
                        )
                        dpg.add_button(
                            label="5s >>",
                            tag="video_skip_forward_button",
                            callback=owner._skip_actor_video,
                            user_data=5.0,
                            enabled=False,
                            width=78,
                        )
                        dpg.add_text("00:00 / 00:00", tag="video_playback_time")
                        dpg.add_checkbox(
                            label="Mostrar landmarks",
                            tag="video_show_landmarks",
                            default_value=True,
                            callback=owner._on_video_landmarks_changed,
                        )


def build_support_windows(dpg, owner) -> None:
    add_help_modal(dpg, owner)
