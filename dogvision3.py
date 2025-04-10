import pygame  # This helps us make a window and show pictures
import cv2  # This lets us use the camera and change pictures
import numpy as np  # This helps us do math with lots of numbers at once

# Start pygame so we can use it to show stuff on the screen
pygame.init()

# Turn on the camera (the "0" means use the first camera the computer finds)
cap = cv2.VideoCapture(0)

# Check if the camera turned on okay
if not cap.isOpened():
    print("Oops! The camera didn’t turn on.")
    exit()  # Stop the program if the camera doesn’t work

# Find out how big the camera’s pictures are
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # How wide the picture is
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # How tall the picture is

# Make a full-screen window that matches the camera’s picture size
screen = pygame.display.set_mode((frame_width, frame_height), pygame.FULLSCREEN)

# This is like a timer to keep the pictures moving smoothly (not too fast)
clock = pygame.time.Clock()

# This is a switch to turn the dog vision on or off (starts ON)
dog_vision_enabled = True

# Make a font (like a style for letters) to write words on the screen
font = pygame.font.Font(None, 36)  # 36 is the size of the letters

def apply_dog_vision_filter(frame):
    """
    This function changes the picture to look like what a dog sees!
    Dogs only see blue and yellow well, not red or green, so we make those special.
    """
    # Change the picture to a color system called HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the picture into its 3 parts: Hue (h), Saturation (s), and Value (v)
    h, s, v = cv2.split(hsv)

    # Make the saturation (color strength) a number we can stretch without breaking
    s = s.astype(np.float32)

    # Dogs love blue! Find blue parts (100 to 140 in hue) and make them stronger
    blue_mask = (h >= 100) & (h <= 140)
    s[blue_mask] = np.minimum(s[blue_mask] * 1.5, 255)  # Boost blue by 1.5x

    # Dogs also like yellow! Find yellow parts (20 to  꿈 in hue) and boost them
    yellow_mask = (h >= 20) & (h <= 40)
    s[yellow_mask] = np.minimum(s[yellow_mask] * 1.5, 255)  # Boost yellow by 1.5x

    # Dogs don’t see red or green well, so we make those boring (almost gray)
    red_green_mask = (h < 20) | ((h > 40) & (h < 100))
    s[red_green_mask] = np.maximum(s[red_green_mask] * 0.1, 0)  # Almost no color here

    # Other colors get a little weaker too (not blue or yellow)
    non_blue_yellow_mask = ~(blue_mask | yellow_mask | red_green_mask)
    s[non_blue_yellow_mask] = np.maximum(s[non_blue_yellow_mask] * 0.5, 0)

    # Change the saturation back to a number the picture understands
    s = s.astype(np.uint8)

    # Put the 3 parts (h, s, v) back together into one picture
    hsv = cv2.merge([h, s, v])

    # Change the picture back to normal colors (BGR) so we can show it
    filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return filtered_frame  # Give back the new dog-vision picture

# Keep going until we say stop!
try:
    while True:
        # Grab a picture from the camera
        ret, frame = cap.read()

        # If we didn’t get a picture, say oops and stop
        if not ret:
            print("Oops! Couldn’t get a picture from the camera.")
            break

        # Turn the picture 90 degrees counterclockwise to make it upright
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Flip the frame horizontally (left becomes right, right becomes left)
        frame = cv2.flip(frame, 1)

        # If dog vision is ON, we’ll only change the LEFT half of the picture
        if dog_vision_enabled:
            # Find the middle of the picture’s height (split it left and right after rotation)
            middle = frame.shape[0] // 2  # frame.shape[0] is the height after rotation

            # Cut the picture into left and right pieces (based on height because of rotation)
            left_half = frame[:middle, :]  # From top to middle, all the way across
            right_half = frame[middle:, :]  # From middle to bottom, all the way across

            # Change only the left half to dog vision
            right_half_dog = apply_dog_vision_filter(right_half)

            # Stick the dog-vision left half and normal right half back together
            frame = np.vstack((left_half, right_half_dog))  # Stack them up and down

        # The camera gives us BGR colors, but pygame likes RGB, so switch them
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Turn the picture into something pygame can show
        frame_surface = pygame.surfarray.make_surface(frame_rgb)

        # Put the picture on the screen
        screen.blit(frame_surface, (0, 0))

        # Write on the screen if the dog filter is ON or OFF
        filter_status = "Dog Filter: ON" if dog_vision_enabled else "Dog Filter: OFF"
        text_surface = font.render(filter_status, True, (255, 255, 255))  # White letters
        # Put the text in the top-right corner (10 pixels from the edge)
        screen.blit(text_surface, (frame_width - text_surface.get_width() - 10, 10))

        # Show the new picture on the screen
        pygame.display.flip()

        # Listen for what the person does (like pressing keys or tapping the screen)
        for event in pygame.event.get():
            # If they click the X button or press Q, stop the program
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                raise SystemExit  # This jumps us out to clean up

            # If they press a key, check which one
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:  # Press 1 to turn OFF dog vision
                    dog_vision_enabled = False
                elif event.key == pygame.K_2:  # Press 2 to turn ON dog vision
                    dog_vision_enabled = True

            # If they tap the screen, check where they tapped
            if event.type == pygame.MOUSEBUTTONDOWN:
                tap_x, tap_y = event.pos  # Get where they tapped
                if tap_x < frame_width // 2:  # Left half of the screen turns OFF
                    dog_vision_enabled = False
                else:  # Right half of the screen turns ON
                    dog_vision_enabled = True

        # Don’t go too fast—wait a tiny bit so it’s smooth (30 pictures per second)
        #clock.tick(30)

# This part makes sure we clean up nicely when we’re done
finally:
    cap.release()  # Turn off the camera
    pygame.quit()  # Close the window and stop pygame
