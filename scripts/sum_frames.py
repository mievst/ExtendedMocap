import os

import cv2


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def sum_frames_in_folder(folder_path):
    total_frames = 0
    for file in os.listdir(folder_path):
        if file.endswith(
            (".mp4", ".avi", ".mov")
        ):  # Добавьте или измените форматы файлов по необходимости
            video_path = os.path.join(folder_path, file)
            total_frames += count_frames(video_path)
    return total_frames


folder_path = "./data/mocap/renders"
print(f"Общее количество кадров в видео файлах папки: {sum_frames_in_folder(folder_path)}")
