from .widgets import add_section_header


def add_video_analysis_tab(dpg, owner) -> None:
    with dpg.tab(label="Videos"):
        add_section_header(dpg, owner, "Analisis de actores", "videos")
        with dpg.tab_bar(tag="video_actor_tabs"):
            with dpg.tab(label="Archivo"):
                dpg.add_button(label="Elegir video local", tag="choose_video_button", callback=owner._show_video_file_dialog, width=-1)
                dpg.add_text("Ningun video seleccionado", tag="video_file_text", wrap=300, color=(190, 200, 210))
            with dpg.tab(label="YouTube"):
                dpg.add_input_text(
                    tag="youtube_url_input",
                    hint="https://www.youtube.com/watch?v=...",
                    width=-1,
                )
                dpg.add_text("Descarga local en cache/videos.", tag="youtube_help_text", wrap=300, color=(190, 200, 210))
            with dpg.tab(label="Ajustes"):
                dpg.add_text("Muestreo entre inferencias (seg)")
                dpg.add_slider_float(
                    label="",
                    tag="video_sample_seconds",
                    default_value=0.35,
                    min_value=0.15,
                    max_value=1.25,
                    format="%.2f",
                    width=-1,
                )
                dpg.add_text("Similitud minima para aceptar famoso")
                dpg.add_slider_float(
                    label="",
                    tag="video_min_similarity",
                    default_value=0.34,
                    min_value=0.20,
                    max_value=0.70,
                    format="%.2f",
                    width=-1,
                )
            with dpg.tab(label="Resultados"):
                dpg.add_text("Sin analisis todavia.", tag="video_results_text", wrap=300, color=(225, 230, 235))
        dpg.add_spacer(height=10)
        dpg.add_button(label="Reproducir y reconocer", tag="analyze_video_button", callback=owner._analyze_selected_video, width=-1)
        dpg.add_button(label="Volver al vivo", tag="live_video_button", callback=owner._return_to_live_video, width=-1)
