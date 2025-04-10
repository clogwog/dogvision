import pygame
import cv2
import numpy as np

# Initialize pygame
pygame.init()

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the camera's frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up full-screen display using pygame
screen = pygame.display.set_mode((frame_width, frame_height), pygame.FULLSCREEN)

# Set frame rate
clock = pygame.time.Clock()

# Flag to control the dog vision filter
dog_vision_enabled = True

# Define font for overlay text
font = pygame.font.Font(None, 36)

def apply_dog_vision_filter(frame):
    """
    Apply a color filter that simulates what a dog might see.
    Dogs are dichromatic and have limited color vision, with a preference for blue and yellow.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Boost blue (around 120 degrees) and yellow (around 30 degrees) hues
    blue_mask = (h >= 100) & (h <= 140)  # Blue in HSV hue space
    yellow_mask = (h >= 20) & (h <= 40)  # Yellow in HSV hue space

    # Increase saturation for blue and yellow areas
    s[blue_mask] = np.minimum(s[blue_mask] * 1.5, 255)  # Increase saturation
    s[yellow_mask] = np.minimum(s[yellow_mask] * 1.5, 255)

    # Decrease saturation for red/green areas
    non_blue_yellow_mask = ~(blue_mask | yellow_mask)
    s[non_blue_yellow_mask] = np.maximum(s[non_blue_yellow_mask] * 0.5, 0)  # Lower saturation for non-blue/yellow

    # Merge the modified channels back into HSV
    hsv = cv2.merge([h, s, v])

    # Convert back to BGR color space
    filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return filtered_frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Rotate the frame by 90 degrees counter-clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply the dog vision filter if enabled
    if dog_vision_enabled:
        frame = apply_dog_vision_filter(frame)

    # Convert the frame to RGB (OpenCV captures in BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to a pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    
    # Display the frame on the screen
    screen.blit(frame_surface, (0, 0))

    # Render the overlay text indicating whether the dog filter is on or off
    filter_status = "Dog Filter: ON" if dog_vision_enabled else "Dog Filter: OFF"
    text_surface = font.render(filter_status, True, (255, 255, 255))  # White text
    screen.blit(text_surface, (frame_width - text_surface.get_width() - 10, 10))  # Position at top-right corner

    pygame.display.flip()

    # Handle events to exit or toggle filter
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            cap.release()
            pygame.quit()
            exit()

        # Check for keypresses
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:  # Press '1' to turn off dog vision filter
                dog_vision_enabled = False
            elif event.key == pygame.K_2:  # Press '2' to turn on dog vision filter
                dog_vision_enabled = True

    # Limit the frame rate
    clock.tick(30)  # Adjust to your desired frame rate (30 FPS here)

