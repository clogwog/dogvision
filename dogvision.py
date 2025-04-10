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
    # For blue: we can emphasize hues near 120
    # For yellow: we can emphasize hues near 30
    # We will leave the saturation and value relatively unchanged to prevent darkening
    # Emphasize blue and yellow by increasing the saturation where these colors appear

    # Create a mask for blue and yellow
    blue_mask = (h >= 100) & (h <= 140)  # Blue in HSV hue space
    yellow_mask = (h >= 20) & (h <= 40)  # Yellow in HSV hue space

    # Increase saturation for blue and yellow areas
    s[blue_mask] = np.minimum(s[blue_mask] * 1.5, 255)  # Increase saturation
    s[yellow_mask] = np.minimum(s[yellow_mask] * 1.5, 255)

    # Decrease saturation for red/green areas (suppress reds and greens)
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

    # Apply the dog vision filter
    dog_vision_frame = apply_dog_vision_filter(frame)

    # Convert the frame to RGB (OpenCV captures in BGR by default)
    frame_rgb = cv2.cvtColor(dog_vision_frame, cv2.COLOR_BGR2RGB)
    
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

