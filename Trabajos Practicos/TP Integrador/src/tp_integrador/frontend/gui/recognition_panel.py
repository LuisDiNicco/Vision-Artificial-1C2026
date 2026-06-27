from .widgets import add_section_header


def add_recognition_tab(dpg, owner) -> None:
    with dpg.tab(label="Reconocer"):
        add_section_header(dpg, owner, "Reconocimiento en vivo", "recognition")
        dpg.add_button(label="Modo reconocimiento", tag="recognition_mode_button", callback=owner._activate_recognition_mode, width=-1)
        dpg.add_spacer(height=8)
        dpg.add_checkbox(label="Espejar video", tag="mirror_video", default_value=False)

        dpg.add_separator()
        add_section_header(dpg, owner, "Doble famoso", "celebrity")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Buscar doble", tag="celebrity_button", callback=owner._request_celebrity_search, width=150)
            dpg.add_button(label="Cachear famosos", tag="celebrity_cache_button", callback=owner._request_celebrity_cache, width=150)
        dpg.add_text("Cache: no cargado", tag="celebrity_status_text", wrap=300, color=(190, 200, 210))
