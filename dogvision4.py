import pygame  # This helps us make a window and show pictures
import cv2  # This lets us use the camera and change pictures
import numpy as np  # This helps us do math with lots of numbers at once
import threading
import sys

# Start pygame so we can use it to show stuff on the screen
pygame.init()

# Turn on the camera (the "0" means use the first camera the computer finds)
cap = cv2.VideoCapture(0)

# Check if the camera turned on okay
if not cap.isOpened():
    print("Oops! The camera didn’t turn on.")
    exit()

# Find out how big the camera’s pictures are
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Make a full-screen window that matches the camera’s picture size
screen = pygame.display.set_mode((frame_width, frame_height), pygame.FULLSCREEN)

# This is like a timer to keep the pictures moving smoothly
clock = pygame.time.Clock()

# Modes: 1 - Full Human, 2 - Full Dog Vision, 3 - Split View
mode = 3  # Default to split view

# Make a font to write words on the screen
font = pygame.font.Font(None, 36)

def apply_dog_vision_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32)
    blue_mask = (h >= 100) & (h <= 140)
    s[blue_mask] = np.minimum(s[blue_mask] * 1.5, 255)
    yellow_mask = (h >= 20) & (h <= 40)
    s[yellow_mask] = np.minimum(s[yellow_mask] * 1.5, 255)
    red_green_mask = (h < 20) | ((h > 40) & (h < 100))
    s[red_green_mask] = np.maximum(s[red_green_mask] * 0.1, 0)
    non_blue_yellow_mask = ~(blue_mask | yellow_mask | red_green_mask)
    s[non_blue_yellow_mask] = np.maximum(s[non_blue_yellow_mask] * 0.5, 0)
    s = s.astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def read_keyboard_input():
    global mode
    while True:
        key = sys.stdin.read(1)
        if key == '1':
            mode = 1
        elif key == '2':
            mode = 2
        elif key == '3':
            mode = 3

# Start background thread for keyboard input if running via SSH
threading.Thread(target=read_keyboard_input, daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Oops! Couldn’t get a picture from the camera.")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        
        if mode == 2:
            frame = apply_dog_vision_filter(frame)
        elif mode == 3:
            middle = frame.shape[0] // 2
            left_half = frame[:middle, :]
            right_half = frame[middle:, :]
            right_half_dog = apply_dog_vision_filter(right_half)
            frame = np.vstack((left_half, right_half_dog))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        screen.blit(frame_surface, (0, 0))
        
        mode_text = "Human Vision" if mode == 1 else "Dog Vision" if mode == 2 else "Split View"
        text_surface = font.render(f"Mode: {mode_text}", True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN:
                mode = (mode % 3) + 1
finally:
    cap.release()
    pygame.quit()
