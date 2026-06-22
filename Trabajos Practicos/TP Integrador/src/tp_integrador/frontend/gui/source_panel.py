from .widgets import add_section_header


def add_source_tab(dpg, owner) -> None:
    with dpg.tab(label="Fuente"):
        add_section_header(dpg, owner, "Entrada en vivo", "source")
        dpg.add_radio_button(
            ("Webcam", "Pantalla", "Ventana"),
            default_value=owner._source_label(owner.video_source),
            callback=owner._on_source_changed,
            horizontal=True,
            tag="source_radio",
        )
        dpg.add_text("Ventana")
        dpg.add_combo([], tag="window_combo", width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Aplicar ventana", tag="window_button", callback=owner._open_selected_window, width=150)
            dpg.add_button(label="Refrescar", tag="refresh_windows_button", callback=owner._populate_window_options, width=150)
        dpg.add_text("Camara")
        dpg.add_combo([], tag="camera_combo", width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Cambiar camara", tag="switch_camera_button", callback=owner._switch_camera, width=150)
            dpg.add_button(label="Siguiente", tag="next_camera_button", callback=owner._next_camera, width=150)
