def add_section_header(dpg, owner, title: str, help_topic: str) -> None:
    with dpg.group(horizontal=True):
        dpg.add_text(title, color=(245, 247, 250))
        dpg.add_button(label="?", width=28, callback=lambda *args, topic=help_topic: owner._show_help(topic))


def add_label_with_tooltip(dpg, label: str, help_text: str) -> None:
    """Etiqueta compacta con ayuda contextual al pasar sobre el icono."""
    with dpg.group(horizontal=True):
        dpg.add_text(label, color=(190, 200, 210))
        help_button = dpg.add_button(label="?", width=24, height=20)
    with dpg.tooltip(help_button):
        dpg.add_text(help_text, wrap=300, color=(225, 230, 235))
