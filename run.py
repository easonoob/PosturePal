import torch
import cv2
import time
import argparse
import requests
import numpy as np
import base64
from io import BytesIO
from openai import OpenAI
from pathlib import Path
import threading
import os
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0) # 1
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.2)#0.7125)
args = parser.parse_args()

height = 180
openai_api_key = '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'
interests = ["AI", "Science", "GPU", "Brawl Stars"]
remind_interval = 30 # seconds

client = OpenAI(
    api_key=openai_api_key,
    organization='org-BzkVKW4kSGfQSRNOmP6h1IU2',
)

def calculate_roll_and_approximate_pitch(keypoint_coords):
    # Extract keypoints
    left_eye = np.array(keypoint_coords[1][:2])
    right_eye = np.array(keypoint_coords[2][:2])
    left_shoulder = np.array(keypoint_coords[5][:2])
    right_shoulder = np.array(keypoint_coords[6][:2])

    # Calculate vectors
    eye_vector = right_eye - left_eye
    shoulder_vector = right_shoulder - left_shoulder

    # Roll: angle of the eye line relative to horizontal
    roll = np.arctan2(eye_vector[1], eye_vector[0]) * (180 / np.pi)

    # Normalize roll to [-180, 180] range
    if roll > 180:
        roll -= 360
    roll += 90

    # Pitch approximation: We'll define pitch based on the distance between the eye line and a point in the middle of the shoulder line
    # This is a heuristic and doesn't directly correspond to pitch but can give some indication of head tilt
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    eye_mid = (left_eye + right_eye) / 2
    mid_vector = eye_mid - shoulder_mid
    pitch = np.arctan2(mid_vector[1], mid_vector[0]) * (180 / np.pi) - roll  # Adjusting by roll for a rough approximation

    # Normalize pitch to make it more intuitive (optional)
    pitch = -pitch # Inverting to match conventional pitch directions
    pitch += 180
    if pitch > 180:
        pitch -= 360
    pitch += height / 90
    # print(roll, pitch)

    return roll, pitch

def play_audio(audio_file):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the audio file
    pygame.mixer.music.load(audio_file)
    # Play the audio
    pygame.mixer.music.play()
    # Keep the thread alive until the music is playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(1)

# Define a thread to play audio without blocking the main program
def start_audio_thread(audio_file):
    audio_thread = threading.Thread(target=play_audio, args=(audio_file,))
    audio_thread.start()

def image_to_base64(image_array, format='jpeg'):
    # Encode the image to the specified format
    image_array = (image_array + 1) * 127.5
    image_array = np.transpose(image_array.squeeze(0), (1, 2, 0))
    _, buffer = cv2.imencode(f'.{format}', image_array)
    
    # Convert the buffer to a byte stream
    byte_stream = BytesIO(buffer)
    
    # Encode the byte stream to base64
    base64_encoded_image = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
    
    return base64_encoded_image

n_audios = 0

def api(raw_image: list, status: str, head_angles: tuple):
    roll, pitch = head_angles
    base64_image = image_to_base64(raw_image, format='jpeg')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    if status == 'good':
        gay = "in a good position"
    elif status == 'bad':
        gay = "in a bad position"

    prompt = f"""You are PosturePal, a friendly and humourous assistant that detects the user's 
        sitting posture and remind him if it is wrong or encourage him to keep it up if it is correct 
        based on the user's interests: {', '.join(interests)} in a FUNNY, CREATIVE, PERSONALIZED, and 
        HUMOUROUS way. The image shows what the webcam captured. PoseNet is used to calculate the angles 
        of the head: head roll {roll:.2f} degrees, head pitch {pitch:.2f} degrees. Use the result by PoseNet 
        with caution since PoseNet is not very accurate. According to the head angles, the user is 
        currectly sitting {gay} (this might be wrong, refer to the image for accuracy). Remind the 
        user directly, say to him directly. Do NOT mention PoseNet, head angle, de NOT be childish, 
        and keep the response in a few sentences. Do NOT refuse to provide feedback based on the image. 
        Also do not keep telling the user the same topic, tell some topics not in the user's interests, 
        and tell some jokes too!"""

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{"role": "user",
                      "content": [{"type": "text", "text": prompt},
                                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                  ]
                    }],
        "max_tokens": 50,
        "temperature": 1.0,
    }
    response_text = "sorry"
    while "sorry" in response_text or "cannot provide" in response_text or "can't assist" in response_text or "cannot assist" in response_text or "can't provide" in response_text or "can't help" in response_text or "cannot help" in response_text:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print("Error:", response.json())
            return
        response_text = response.json()['choices'][0]['message']['content']
    print(response_text)

    global n_audios
    speech_file_path = Path(__file__).parent / "tts" / f"speech{n_audios}.mp3"
    audio_payload = {
        "model": "tts-1-hd",
        "voice": "fable",
        "input": response_text,
    }
    audio_response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=audio_payload)
    # audio_response = client.audio.speech.create(
    #     model="tts-1-hd",
    #     voice="alloy",
    #     input=response
    # )
    # audio_response.with_streaming_response.method(speech_file_path)
    if audio_response.status_code == 200:
        # os.remove(speech_file_path) if os.path.exists(speech_file_path) else None
        os.makedirs("tts") if not os.path.exists("tts") else None
        with open(speech_file_path, 'wb') as audio_file:
            audio_file.write(audio_response.content)
    else:
        print(f"Error: {audio_response.status_code}")
        print(audio_response.text)

    start_audio_thread(speech_file_path)
    n_audios += 1

def main():
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    roll_history = []
    pitch_history = []
    good_position_time = 0
    last_time = time.time()
    last_request_time = time.time()
    best_time = 0
    average = lambda lst: sum(lst)/len(lst)
    
    while True:
        input_image_raw, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image_raw)#.cuda()
            input_image = (input_image + 1) / 2 # normalization

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords, _ = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.2)
        
        for pi, (pose_score, keypoint_score, keypoint_coord) in enumerate(zip(pose_scores, keypoint_scores, keypoint_coords)):
            if pose_score > 0.15:
                roll, pitch = calculate_roll_and_approximate_pitch(keypoint_coord)
                roll_history.append(roll)
                pitch_history.append(pitch)

        if frame_count % 10 == 0 and frame_count != 0:
            average_roll = average(roll_history[-10:])
            average_pitch = average(pitch_history[-10:])
            print(f"Pose {pi}: Average Roll: {average_roll:.2f} degrees, Average Pitch: {average_pitch:.2f} degrees")
            if average_roll <= 20 and average_roll >= -20 and average_pitch <= 10 and average_pitch >= -10:
                good_position_time += time.time() - last_time
                last_time = time.time()
                best_time = max(best_time, good_position_time)
                print(f"Good Position Time: {good_position_time:.2f}s, Best Time: {best_time:.2f}s")
                if time.time() - last_request_time > remind_interval*5:
                    api(input_image_raw, "good", (average_roll, average_pitch))
                    last_request_time = time.time()
            else:
                good_position_time = 0
                last_time = time.time()
                print("You lost the good position streak!")
                if time.time() - last_request_time > remind_interval:
                    api(input_image_raw, "bad", (average_roll, average_pitch))
                    last_request_time = time.time()

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()