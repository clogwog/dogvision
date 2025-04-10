import pygame
import cv2
import numpy as np

pygame.init()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Oops! The camera didn’t turn on.")
    exit()

# Get camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get screen resolution
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

clock = pygame.time.Clock()
dog_vision_enabled = True
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
    
    s = s.astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Oops! Couldn’t get a picture from the camera.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)

        middle = frame.shape[1] // 2  # Split using width
        left_half, right_half = frame[:, :middle], frame[:, middle:]
        if dog_vision_enabled:
            right_half_dog = apply_dog_vision_filter(right_half)
            frame = np.hstack((left_half, right_half_dog))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
        screen.blit(frame_surface, (0, 0))

        filter_status = "Dog Filter: ON" if dog_vision_enabled else "Dog Filter: OFF"
        text_surface = font.render(filter_status, True, (255, 255, 255))
        screen.blit(text_surface, (screen_width - text_surface.get_width() - 10, 10))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    dog_vision_enabled = False
                elif event.key == pygame.K_2:
                    dog_vision_enabled = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                tap_x, _ = event.pos
                dog_vision_enabled = tap_x >= screen_width // 2

finally:
    cap.release()
    pygame.quit()

