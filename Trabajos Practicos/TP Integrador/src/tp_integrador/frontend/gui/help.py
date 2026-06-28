HELP_TOPICS = {
    "general": (
        "Ayuda",
        "Selecciona un flujo en las pestanas del panel lateral y usa el panel de video para validar que la cara este bien visible.",
    ),
    "training": (
        "Entrenar personas",
        "1. Escribi nombre y apellido. 2. Entra en modo entrenamiento. 3. Captura varias muestras con buena luz, cara grande y pose frontal. 4. Repeti para cada persona y entrena el SVM.",
    ),
    "recognition": (
        "Reconocimiento en vivo",
        "Activa el modo reconocimiento despues de entrenar el SVM. La etiqueta aparece sobre cada rostro detectado. Si la confianza baja, suma mas capturas de entrenamiento con mejor calidad.",
    ),
    "videos": (
        "Analisis de videos",
        "Elegi un archivo local o pega una URL de YouTube. El sistema muestrea frames, descarta caras de baja calidad, agrupa apariciones y compara embeddings promedio contra famosos.",
    ),
    "source": (
        "Fuente en vivo",
        "La app usa solo la webcam de OpenCV. Si la camara no responde, cambia el indice desde el panel lateral y presiona Cambiar camara.",
    ),
}


def add_help_modal(dpg, owner) -> None:
    with dpg.window(
        tag="help_modal",
        label="Ayuda",
        modal=True,
        show=False,
        no_resize=True,
        width=560,
        height=320,
    ):
        dpg.add_text("", tag="help_title", color=(120, 210, 255))
        dpg.add_separator()
        dpg.add_text("", tag="help_body", wrap=520, color=(225, 230, 235))
        dpg.add_spacer(height=14)
        dpg.add_button(label="Cerrar", callback=owner._hide_help, width=-1)
