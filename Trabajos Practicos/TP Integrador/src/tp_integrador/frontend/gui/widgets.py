def add_section_header(dpg, owner, title: str, help_topic: str) -> None:
    with dpg.group(horizontal=True):
        dpg.add_text(title, color=(245, 247, 250))
        dpg.add_button(label="?", width=28, callback=lambda *args, topic=help_topic: owner._show_help(topic))
