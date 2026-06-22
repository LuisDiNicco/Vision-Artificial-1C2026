from .widgets import add_section_header


def add_training_tab(dpg, owner) -> None:
    with dpg.tab(label="Entrenar"):
        add_section_header(dpg, owner, "1. Persona", "training")
        dpg.add_input_text(
            tag="person_name",
            hint="Nombre y apellido",
            width=-1,
            callback=owner._on_person_changed,
        )
        dpg.add_text("Embeddings guardados: 0", tag="person_count_text", color=(160, 210, 180))
        dpg.add_checkbox(label="Guardar fotos alineadas", tag="save_photos", default_value=False)

        dpg.add_spacer(height=8)
        dpg.add_text("2. Capturas")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Modo entrenamiento", tag="training_mode_button", callback=owner._activate_training_mode, width=150)
            dpg.add_button(label="Capturar", tag="capture_button", callback=owner._request_capture, width=150)

        dpg.add_spacer(height=8)
        dpg.add_text("3. Modelo")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Entrenar SVM", tag="train_button", callback=owner._request_train, width=150)
            dpg.add_button(label="Nueva persona", tag="new_person_button", callback=owner._new_person, width=150)
