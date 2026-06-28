from .widgets import add_section_header


def add_recognition_tab(dpg, owner) -> None:
    with dpg.tab(label="Reconocer"):
        add_section_header(dpg, owner, "Reconocimiento en vivo", "recognition")
        dpg.add_button(label="Modo reconocimiento", tag="recognition_mode_button", callback=owner._activate_recognition_mode, width=-1)
        dpg.add_spacer(height=8)
        dpg.add_checkbox(label="Espejar video", tag="mirror_video", default_value=False)
