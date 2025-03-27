import flet as ft
from backend import PostureBackend

def main(page: ft.Page):
    page.title = "PostureSeeker"
    page.window_width = 1000
    page.window_height = 700

    # We store whether detection is running
    detection_running = True

    # Create placeholders for text, camera image, etc.
    status_text = ft.Text("You are doing great!", size=20)
    score_text = ft.Text("Score: 390 | Good position streak: 1 hour", size=16)
    camera_image = ft.Image(
        src="",
        width=500,
        height=350,
        fit=ft.ImageFit.CONTAIN
    )
    plot_placeholder = ft.Container(
        content=ft.Text("Some Plots Here", size=18),
        alignment=ft.alignment.center,
        bgcolor=ft.colors.AMBER_50,
        width=300,
        height=350
    )

    # -- Settings Modal Controls --
    openai_key_field = ft.TextField(label="OpenAI API Key", width=400)
    camera_id_field = ft.TextField(label="Camera ID", width=400, value="0")
    theme_dropdown = ft.Dropdown(
        width=200,
        options=[
            ft.dropdown.Option("LIGHT"),
            ft.dropdown.Option("DARK"),
        ],
        value="LIGHT"
    )

    def close_settings_modal(e):
        page.theme_mode = (
            ft.ThemeMode.LIGHT if theme_dropdown.value == "LIGHT" else ft.ThemeMode.DARK
        )
        # Apply new camera ID and openai key to backend
        if camera_id_field.value.isdigit():
            backend.set_camera_id(int(camera_id_field.value))
        backend.set_openai_api_key(openai_key_field.value)

        # settings_modal.open = False
        page.close(settings_modal)
        print("Settings saved")
        page.update()

    settings_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("Settings"),
        content=ft.Column(
            [
                openai_key_field,
                camera_id_field,
                ft.Row([ft.Text("Theme:"), theme_dropdown]),
            ],
            tight=True,
        ),
        actions=[
            ft.ElevatedButton("Close", on_click=close_settings_modal),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    # Add modal to the page
    page.dialog = settings_modal

    # -- Backend object --
    def update_image_callback(frame_b64):
        camera_image.src_base64 = frame_b64
        page.update()

    def update_text_callback(posture_str, score_val, streak_val):
        status_text.value = posture_str
        score_text.value = f"Score: {score_val} | Good position streak: {streak_val} hour"
        page.update()

    backend = PostureBackend(update_image_callback, update_text_callback)

    # Start detection right away
    backend.start_detection()

    # -- Button handlers --
    def toggle_detection(e):
        nonlocal detection_running
        if detection_running:
            # Stop detection
            backend.stop_detection()
            detection_running = False
            detection_button.text = "Start Detection"
        else:
            # Start detection
            backend.start_detection()
            detection_running = True
            detection_button.text = "Stop Detection"
        page.update()

    # def open_settings(e):
    #     # Show modal
    #     settings_modal.open = True
    #     page.update()

    detection_button = ft.ElevatedButton("Stop Detection", on_click=toggle_detection)
    settings_button = ft.ElevatedButton("Settings", on_click=lambda e: page.open(settings_modal))

    # --- Layout (2 rows, each row with 2 columns) ---
    # Row 1: [ Col1(buttons), Col2(status text) ]
    # Row 2: [ camera image, plots ]

    top_row = ft.Row(
        [
            ft.Column(
                [
                    detection_button,
                    settings_button,
                ],
                width=200,
                alignment=ft.MainAxisAlignment.START
            ),
            ft.Column(
                [
                    status_text,
                    score_text,
                ],
                alignment=ft.MainAxisAlignment.START
            )
        ],
        alignment=ft.MainAxisAlignment.START
    )

    bottom_row = ft.Row(
        [
            camera_image,
            plot_placeholder
        ],
        alignment=ft.MainAxisAlignment.START
    )

    page.add(
        top_row,
        bottom_row
    )

ft.app(target=main)
