from .help import add_help_modal
from .recognition_panel import add_recognition_tab
from .source_panel import add_source_tab
from .training_panel import add_training_tab
from .video_panel import add_video_analysis_tab


def build_main_window(dpg, owner, video_w: int, video_h: int) -> None:
    with dpg.window(tag="main_window", label="TP Integrador - Reconocimiento Facial", no_close=True):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="sidebar_panel", width=340, height=820, border=True):
                dpg.add_text("TP Integrador", tag="title_text", color=(245, 247, 250))
                dpg.add_text("Reconocimiento facial", tag="subtitle_text", color=(120, 210, 255))
                dpg.add_separator()

                with dpg.tab_bar(tag="workflow_tabs"):
                    add_training_tab(dpg, owner)
                    add_recognition_tab(dpg, owner)
                    add_video_analysis_tab(dpg, owner)
                    add_source_tab(dpg, owner)

                dpg.add_separator()
                dpg.add_text("Estado")
                dpg.add_text("", tag="status_text", wrap=300, color=(225, 230, 235))
                dpg.add_text("", tag="stats_text", wrap=300, color=(160, 170, 180))

            with dpg.child_window(tag="video_panel", width=1308, height=820, border=False):
                dpg.add_image("video_texture", tag="video_image", width=video_w, height=video_h)


def add_video_file_dialog(dpg, owner) -> None:
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=owner._on_video_file_selected,
        tag="video_file_dialog",
        width=720,
        height=460,
    ):
        dpg.add_file_extension(".mp4", color=(90, 210, 255, 255))
        dpg.add_file_extension(".avi", color=(90, 210, 255, 255))
        dpg.add_file_extension(".mov", color=(90, 210, 255, 255))
        dpg.add_file_extension(".mkv", color=(90, 210, 255, 255))


def build_support_windows(dpg, owner) -> None:
    add_video_file_dialog(dpg, owner)
    add_help_modal(dpg, owner)
