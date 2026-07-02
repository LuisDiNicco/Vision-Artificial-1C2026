from .widgets import add_section_header


def add_source_tab(dpg, owner) -> None:
    with dpg.tab(label="Fuente"):
        add_section_header(dpg, owner, "Entrada en vivo", "source")
        dpg.add_text("Camara")
        dpg.add_combo([], tag="camera_combo", width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Cambiar camara", tag="switch_camera_button", callback=owner._switch_camera, width=150)
            dpg.add_button(label="Siguiente", tag="next_camera_button", callback=owner._next_camera, width=150)
