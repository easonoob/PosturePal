import cv2
import torch
import numpy as np
import base64
import time
import threading
import requests
from pathlib import Path
from io import BytesIO
import posenet
import pygame

class PostureBackend:
    def __init__(self, update_image_callback, update_text_callback):
        self.update_image_callback = update_image_callback
        self.update_text_callback = update_text_callback

        self.model = posenet.load_model(6969)
        self.resnet = posenet.get_resnet(180)
        self.output_stride = self.model.output_stride

        self.camera_id = 0
        self.api_key = ""
        self.interests = ["AI", "Science", "Large Language Models (LLM)"]
        self.remind_interval = 1800
        self.n_audios = 0
        self.stop_flag = False
        self.detection_thread = None

        pygame.mixer.init()

    def set_camera_id(self, cam_id):
        self.camera_id = cam_id

    def set_openai_api_key(self, api_key):
        self.api_key = api_key

    def encode_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def play_audio(self, audio_file):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(1)

    def start_detection(self):
        if self.detection_thread is not None and self.detection_thread.is_alive():
            return
        self.stop_flag = False
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        self.stop_flag = True
        if self.detection_thread:
            self.detection_thread.join()

    def api_request(self, raw_image, status, head_angles, best_time):
        roll, pitch = head_angles
        base64_image = self.encode_frame(raw_image)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        prompt = f"""You are PosturePal, a friendly assistant for posture detection. Provide personalized and humorous feedback.
        Head Roll: {roll:.2f}, Head Pitch: {pitch:.2f}. Status: {status}, Best Good Posture Time: {best_time}s."""

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "max_tokens": 150,
            "temperature": 1.3,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print("Error:", response.json())
            return

        response_text = response.json()['choices'][0]['message']['content']
        print(response_text)

        audio_file_path = Path(f"tts/speech{self.n_audios}.mp3")
        audio_payload = {
            "model": "tts-1-hd",
            "voice": "fable",
            "input": response_text,
        }
        audio_response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=audio_payload)

        if audio_response.status_code == 200:
            Path("tts").mkdir(exist_ok=True)
            with open(audio_file_path, 'wb') as f:
                f.write(audio_response.content)
            threading.Thread(target=self.play_audio, args=(audio_file_path,)).start()
            self.n_audios += 1

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Failed to open camera.")
            return

        frame_count = 0
        roll_history, pitch_history = [1], [1]
        good_position_time, best_time = 0, 0
        last_time = last_request_time = time.time()

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            input_image_raw, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.2, output_stride=self.output_stride)
            input_image = torch.Tensor(input_image_raw)

            with torch.no_grad():
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.model(input_image)
                pose_scores, keypoint_scores, keypoint_coords, _ = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=self.output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.2)

                for pose_score, keypoint_score, keypoint_coord in zip(pose_scores, keypoint_scores, keypoint_coords):
                    if pose_score > 0.15:
                        roll, pitch = self.resnet(input_image, keypoint_coord)
                        roll_history.append(roll)
                        pitch_history.append(pitch)

            avg_roll = sum(roll_history[-10:]) / len(roll_history[-10:])
            avg_pitch = sum(pitch_history[-10:]) / len(pitch_history[-10:])

            if -20 <= avg_roll <= 20 and -10 <= avg_pitch <= 10:
                good_position_time += time.time() - last_time
                best_time = max(best_time, good_position_time)
                status = 'good'
            else:
                good_position_time = 0
                status = 'bad'

            if time.time() - last_request_time > 60:
                self.api_request(frame, status, (avg_roll, avg_pitch), best_time)
                last_request_time = time.time()

            self.update_image_callback(self.encode_frame(frame))
            self.update_text_callback("You are doing great!" if status == 'good' else "Adjust your posture!", 999, int(best_time / 3600))

            last_time = time.time()
            frame_count += 1

        cap.release()
        print('Detection stopped.')
