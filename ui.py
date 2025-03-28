import flet as ft
from backend import PostureBackend

historical_scores = []
temp_historical_scores = []

def main(page: ft.Page):
    page.title = "PostureSeeker"
    page.window_width = 1000
    page.window_height = 700
    page.theme_mode = (ft.ThemeMode.DARK)

    detection_running = True

    status_text = ft.Text("You are doing great!", size=20)
    score_text = ft.Text("Score: 390 | Good position streak: 1 Minutes", size=16)
    camera_image = ft.Image(
        src="",
        width=500,
        height=350,
        fit=ft.ImageFit.CONTAIN
    )
    # plot_placeholder = ft.Container(
    #     content=ft.Text("Some Plots Here", size=18),
    #     alignment=ft.alignment.center,
    #     bgcolor=ft.Colors.AMBER_50,
    #     width=300,
    #     height=350
    # )
    score_chart = ft.LineChart(
        data_series=[
            ft.LineChartData(
                data_points=[ft.LineChartDataPoint(x, y) for x, y in enumerate(historical_scores)],
                stroke_width=3,
                color=ft.Colors.BLUE_600,
                curved=True,
                stroke_cap_round=True
            )
        ],
        width=450,
        height=350,
        left_axis=ft.ChartAxis(
            labels_size=40,
            title=ft.Text("Score", size=16, weight=ft.FontWeight.BOLD),
            title_size=20,
            labels=[
                ft.ChartAxisLabel(value=0, label=ft.Text("0")),
                ft.ChartAxisLabel(value=10, label=ft.Text("10")),
                ft.ChartAxisLabel(value=20, label=ft.Text("20")),
                ft.ChartAxisLabel(value=30, label=ft.Text("30")),
                ft.ChartAxisLabel(value=40, label=ft.Text("40")),
                ft.ChartAxisLabel(value=50, label=ft.Text("50")),
                ft.ChartAxisLabel(value=60, label=ft.Text("60")),
                ft.ChartAxisLabel(value=70, label=ft.Text("70")),
                ft.ChartAxisLabel(value=80, label=ft.Text("80")),
            ],
            labels_interval=10,  # Show labels every 10 units
            show_labels=True
        ),
        bottom_axis=ft.ChartAxis(
            labels_size=40,
            title=ft.Text("Time (Recent Updates)", size=16, weight=ft.FontWeight.BOLD),
            title_size=20,
            labels_interval=1,
            show_labels=True
        ),
        min_y=0,
        max_y=80,
        # bgcolor=ft.Colors.GREY_400,
        tooltip_bgcolor=ft.Colors.with_opacity(0.9, ft.Colors.BLUE_100)
    )

    score_chart_container = ft.Container(
        content=score_chart,
        border=ft.border.all(1, ft.Colors.GREY_700),
        border_radius=10,
        width=450,
        height=350
    )

    # -- Settings Modal Controls --
    openai_key_field = ft.TextField(label="OpenAI API Key", width=400, multiline=True, min_lines=1, max_lines=3)
    camera_id_field = ft.TextField(label="Camera ID", width=400, value="0")
    theme_dropdown = ft.Dropdown(
        width=200,
        options=[
            ft.dropdown.Option("LIGHT"),
            ft.dropdown.Option("DARK"),
        ],
        value="DARK"
    )
    interest_field = ft.TextField(label="Interest", width=400, value='AI, Science, Large Language Models (LLM)', multiline=True, min_lines=1, max_lines=5)

    def close_settings_modal(e):
        page.theme_mode = (
            ft.ThemeMode.LIGHT if theme_dropdown.value == "LIGHT" else ft.ThemeMode.DARK
        )
        if camera_id_field.value.isdigit():
            backend.set_camera_id(int(camera_id_field.value))
        backend.set_openai_api_key(openai_key_field.value)
        backend.set_interest(interest_field.value)

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
                interest_field,
            ],
            tight=True,
        ),
        actions=[
            ft.ElevatedButton("Close", on_click=close_settings_modal),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    page.dialog = settings_modal

    # -- Backend object --
    def update_image_callback(frame_b64):
        camera_image.src_base64 = frame_b64
        page.update()

    def update_text_callback(posture_str, score_val, streak_val):
        status_text.value = posture_str
        score_text.value = f"Score: {score_val} | Good position streak: {streak_val} minute(s)"

        global temp_historical_scores
        temp_historical_scores.append(score_val)
        if len(temp_historical_scores) > 10:
            # # historical_scores.pop(0)
            # historical_scores.append(sum(temp_historical_scores)/len(temp_historical_scores))
            # temp_historical_scores = []
            # score_chart.data_series[0].data_points = [
            #     ft.LineChartDataPoint(x, y) for x, y in enumerate(historical_scores)
            # ]
            historical_scores.append(sum(temp_historical_scores) / len(temp_historical_scores))
            temp_historical_scores = []

            # Update chart data points
            score_chart.data_series[0].data_points = [
                ft.LineChartDataPoint(x, y) for x, y in enumerate(historical_scores)
            ]

            # Dynamically adjust chart width based on number of data points
            num_points = len(historical_scores)
            chart_width = max(450, num_points * 20)  # 20 pixels per data point, minimum 450
            score_chart.width = chart_width
            score_chart_container.width = chart_width

            # Dynamically adjust x-axis labels_interval to show ~5-10 labels in the visible area
            visible_points = min(num_points, 450 // 20)  # Number of points visible in 450px
            if visible_points > 0:
                desired_labels = min(max(5, visible_points // 10), 10)  # Between 5 and 10 labels
                score_chart.bottom_axis.labels_interval = max(1, visible_points // desired_labels)

            # Auto-scroll to the rightmost position (latest time)
            # if num_points > 0:
            #     # Calculate the scroll offset to show the latest data point
            #     scroll_offset = max(0, chart_width - 450)  # 450 is the visible width
                # score_chart_container.scroll_to(offset=scroll_offset, duration=500)  # Smooth scroll to the end


        page.update()

    backend = PostureBackend(update_image_callback, update_text_callback)

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
            score_chart_container
        ],
        alignment=ft.MainAxisAlignment.START
    )

    page.add(
        top_row,
        bottom_row
    )

ft.app(target=main)
