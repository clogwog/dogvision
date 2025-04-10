import pygame
import cv2
import numpy as np

# Initialize pygame
pygame.init()

# Initialize camera capture
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the camera's frame size
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"width x height: {frame_width} x {frame_height}")

# Set up full-screen display using pygame
screen = pygame.display.set_mode((frame_width, frame_height), pygame.FULLSCREEN)

# Set frame rate
clock = pygame.time.Clock()

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Rotate the frame by 90 degrees counter-clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Convert the frame to RGB (OpenCV captures in BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to a pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    
    # Display the frame on the screen
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # Handle events to exit (e.g., pressing 'q' to quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            cap.release()
            pygame.quit()
            exit()

    # Limit the frame rate
    clock.tick(30)  # Adjust to your desired frame rate (30 FPS here)

