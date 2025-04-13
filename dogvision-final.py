import pygame  # This helps us make a window and show pictures
import cv2  # This lets us use the camera and change pictures
import numpy as np  # This helps us do math with lots of numbers at once

# Start pygame so we can use it to show stuff on the screen
pygame.init()

# Turn on the camera (the "0" means use the first camera the computer finds)
cap = cv2.VideoCapture(0)

# Check if the camera turned on okay
if not cap.isOpened():
    print("Oops! The camera didn't turn on.")
    exit()

# Find out how big the camera's pictures are
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create screen with the correct dimensions
screen = pygame.display.set_mode((frame_width, frame_height), pygame.FULLSCREEN)

# This is like a timer to keep the pictures moving smoothly
clock = pygame.time.Clock()

# Make a font to write words on the screen
font = pygame.font.Font(None, 36)

def apply_dog_vision_filter(frame):
    # Apply blur first to simulate dog's less sharp vision
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # convert it to hue saturation and value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32)

    # select the blue colours
    blue_mask = (h >= 100) & (h <= 140)
    s[blue_mask] = np.minimum(s[blue_mask] * 1.5, 255)

    #select the yellow colours
    yellow_mask = (h >= 20) & (h <= 40)
    s[yellow_mask] = np.minimum(s[yellow_mask] * 1.5, 255)

    #select red and green colour
    red_green_mask = (h < 20) | ((h > 40) & (h < 100))
    s[red_green_mask] = np.maximum(s[red_green_mask] * 0.1, 0)

    # select every other colour
    non_blue_yellow_mask = ~(blue_mask | yellow_mask | red_green_mask)
    s[non_blue_yellow_mask] = np.maximum(s[non_blue_yellow_mask] * 0.5, 0)

    # convert it back to what the format was
    s = s.astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Pre-render text surfaces (moved outside loop since they don't change)
mode_text = "Human Vision                                                                                                       Dog Vision"
outline_positions = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1)
]

# Pre-render all text surfaces
outline_surfaces = []
for dx, dy in outline_positions:
    outline_surface = font.render(mode_text, True, (0, 0, 0))
    outline_surfaces.append((outline_surface, (10 + dx, 10 + dy)))
text_surface = font.render(mode_text, True, (255, 255, 255))

try:
    while True:
        # Limit frame rate to 30 FPS
        clock.tick(30)
        
        ret, frame = cap.read()
        if not ret:
            print("Oops! Couldn't get a picture from the camera.")
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, -1)
        
        middle = int(frame.shape[0] * 0.40)
        left_half = frame[:middle, :]
        right_half = frame[middle:, :]
                
        right_half_dog = apply_dog_vision_filter(right_half)
        
        frame = np.vstack((left_half, right_half_dog))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        screen.blit(frame_surface, (0, 0))
        
        # Draw the dividing line
        pygame.draw.line(screen, (0, 0, 0), (middle-1, 0), (middle-1, frame.shape[1]), 3)
        pygame.draw.line(screen, (0, 0, 0), (middle+1, 0), (middle+1, frame.shape[1]), 3)
        pygame.draw.line(screen, (255, 255, 255), (middle, 0), (middle, frame.shape[1]), 1)
        
        # Use pre-rendered text surfaces
        for surface, pos in outline_surfaces:
            screen.blit(surface, pos)
        screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()

finally:
    cap.release()
    pygame.quit()
