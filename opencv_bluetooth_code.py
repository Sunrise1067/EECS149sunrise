
import numpy as np
import json
import os
import math
import asyncio
from bleak import BleakClient
from pyzbar import pyzbar

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2 

# Bluetooth configurations
HM10_ADDRESS = "60:B6:E1:E1:C6:A6"
UART_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# Bluetooth connection management
class BluetoothManager:
    def __init__(self, address, uuid):
        self.address = address
        self.uuid = uuid
        self.client = None
        self.command_sent = False

    async def connect(self):
        self.client = BleakClient(self.address)
        await self.client.connect()
        print("Bluetooth connected")

    async def send(self, message):
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(self.uuid, message.encode('utf-8'))
            print(f"Bluetooth command sent: {message}")
        else:
            print("Bluetooth not connected")

    async def disconnect(self):
        if self.client:
            await self.client.disconnect()
            print("Bluetooth disconnected")

    def reset_command_state(self):
        self.command_sent = False

# Global Bluetooth Manager instance
bluetooth_manager = BluetoothManager(HM10_ADDRESS, UART_CHARACTERISTIC_UUID)

# Bluetooth function to send commands
def send_bluetooth_command(message):
    async def send_task():
        await bluetooth_manager.send(message)

    asyncio.run(send_task())

# File path for saving and loading color definitions
COLOR_DEFINITIONS_FILE = "color_definitions.json"

# Global variables
color_definitions = {}  # Stores color definitions
hsv_image = None         # HSV format of the camera feed

def pick_color(event, x, y, flags, param):
    global hsv_image, color_definitions
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        pixel = hsv_image[y, x]
        h, s, v = pixel
        if h > 179 - 10:
            upper_h = 179
        else:
            upper_h = h + 10
        if h < 10:
            lower_h = 0
        else:
            lower_h = h - 10
        if s > 255 - 50:
            upper_s = 255
        else:
            upper_s = s + 50
        if s < 50:
            lower_s = 0
        else:
            lower_s = s - 50
        if v > 255 - 50:
            upper_v = 255
        else:
            upper_v = v + 50
        if v < 50:
            lower_v = 0
        else:
            lower_v = v - 50

        # 打印上下界限
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        color_name = input(f"Enter color name (HSV: {pixel}): ")
        if color_name:
            color_definitions[color_name] = (lower_bound.tolist(), upper_bound.tolist())
            print(f"Defined color '{color_name}' successfully! HSV range: {lower_bound} ~ {upper_bound}")


# Save color definitions to file
def save_color_definitions():
    with open(COLOR_DEFINITIONS_FILE, "w") as file:
        json.dump(color_definitions, file)
    print(f"Color definitions saved to {COLOR_DEFINITIONS_FILE}")

# Load color definitions from file
def load_color_definitions():
    global color_definitions
    if os.path.exists(COLOR_DEFINITIONS_FILE):
        with open(COLOR_DEFINITIONS_FILE, "r") as file:
            color_definitions = json.load(file)
        print(f"Loaded color definitions: {color_definitions}")
        return True
    return False

def detect_qr_code(frame, qr_code_size):
    decoded_objects = pyzbar.decode(frame)
    if not decoded_objects:
        print("No QR code detected")
        return frame, None, None, None, None

    for obj in decoded_objects:
        points = obj.polygon
        if len(points) != 4:
            continue

        points = np.array([[p.x, p.y] for p in points])
        points = np.roll(points, 3, axis=0)

        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)] 

        # Draw the QR code borders and calculate midpoints
        for i in range(4):
            start_point = tuple(points[i])
            end_point = tuple(points[(i + 1) % 4])
            color = colors[i]
            cv2.line(frame, start_point, end_point, color, 2)

            # Save midpoints of red and blue borders
            midpoint = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            if i == 2:  # Front border
                red_midpoint = midpoint
            elif i == 0:  # Back border
                blue_midpoint = midpoint

        # Calculate the QR code center
        qr_center = tuple(map(int, np.mean(points, axis=0)))
        cv2.circle(frame, qr_center, 5, (0, 0, 255), -1)  # Mark the center

        # Calculate pixels-to-meters ratio
        p0, p1 = points[0], points[1]  # Use the first edge of the QR code
        pixel_distance = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        ppm = qr_code_size / pixel_distance

        break  # Only process the first QR code

    return frame, qr_center, ppm, blue_midpoint, red_midpoint

# Detect colors and return the centers of the detected target ball and base ball (red)
def detect_colors(frame, hsv_frame, target_color):
    target_ball_center = None
    base_ball_center = None
    max_target_radius = -1
    max_base_radius = -1
    largest_other_balls = {}  # To store the largest "other balls" by color

    for color_name, (lower_bound, upper_bound) in color_definitions.items():
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)

        # Create mask for the current color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Use morphological operations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Detect contours for the current mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small contours
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                if color_name.lower() == target_color:
                    # Highlight the largest target color ball
                    if radius > max_target_radius:
                        max_target_radius = radius
                        target_ball_center = center

                elif color_name.lower() == "dark_green":  # Example: base ball is dark_green
                    # Highlight the largest base ball
                    if radius > max_base_radius:
                        max_base_radius = radius
                        base_ball_center = center

                else:
                    # Track the largest ball for each "other" color
                    if color_name not in largest_other_balls or radius > largest_other_balls[color_name][1]:
                        largest_other_balls[color_name] = (center, radius)

    # Draw the largest target ball
    if target_ball_center:
        cv2.circle(frame, target_ball_center, max_target_radius, (0, 0, 255), 4)
        # cv2.putText(frame, f"Target: {target_color}", 
        #             (target_ball_center[0] - max_target_radius, target_ball_center[1] - max_target_radius - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        print(f"No target ball detected for color: {target_color}")

    # Draw the largest base ball
    if base_ball_center:
        cv2.circle(frame, base_ball_center, max_base_radius, (255, 0, 0), 4)
        cv2.putText(frame, "Base Ball", 
                    (base_ball_center[0] - max_base_radius, base_ball_center[1] - max_base_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        print("No base ball (green) detected")

    # Draw the largest ball for each "other" color
    for color_name, (center, radius) in largest_other_balls.items():
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.putText(frame, color_name, 
                    (center[0] - radius, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, target_ball_center, base_ball_center


# Calculate rotation angle
def calculate_rotation_angle(ball_center, blue_midpoint, red_midpoint):
    if blue_midpoint is None or red_midpoint is None:
        return "Error: Missing midpoints"

    # Calculate angle between the two lines
    angle_num = ((blue_midpoint[0] - red_midpoint[0])/2) ** 2 + ((blue_midpoint[1] - red_midpoint[1])/2) ** 2 + ((blue_midpoint[0] + red_midpoint[0])/2 - ball_center[0]) ** 2 + ((blue_midpoint[1] + red_midpoint[1])/2 - ball_center[1]) ** 2 - (blue_midpoint[0] - ball_center[0]) ** 2 - (blue_midpoint[1] - ball_center[1]) ** 2
    angle_den = 2 * math.sqrt(((blue_midpoint[0] - red_midpoint[0])/2) ** 2 + ((blue_midpoint[1] - red_midpoint[1])/2) ** 2) * math.sqrt(((blue_midpoint[0] + red_midpoint[0])/2 - ball_center[0]) ** 2 + ((blue_midpoint[1] + red_midpoint[1])/2 - ball_center[1]) ** 2)
    angle = math.acos(angle_num / angle_den)
    
    if angle < 0.2:  # Tolerance of 0.09 radians
        return "Aligned"
    else:
        rotation_direction = "R"    # Default rotation direction
        rotation_direction = "L" if (blue_midpoint[0] - ball_center[0]) * (red_midpoint[1] - ball_center[1]) - (blue_midpoint[1] - ball_center[1]) * (red_midpoint[0] - ball_center[0]) > 0 else rotation_direction 
        return f"{rotation_direction} {math.degrees(angle):.1f}"  # Rotation in degrees


# Calculate movement command
def calculate_movement_command(real_distance, threshold_distance=0.075):
    distance_difference = real_distance - threshold_distance
    if abs(distance_difference) <= 0.001:
        return "Aligned"
    elif distance_difference > 0:
        return f"F {abs(distance_difference * 100):.1f}"
    else:
        return f"B {abs(distance_difference * 100):.1f}"
    
# Check if the ball is absorbed based on its proximity to the blue midpoint
def is_ball_absorbed(ball_center, blue_midpoint, ppm, absorption_radius_pixels):
    if ball_center is None or blue_midpoint is None or ppm is None:
        return False
    distance = math.sqrt((ball_center[0] - blue_midpoint[0]) ** 2 +
        (ball_center[1] - blue_midpoint[1]) ** 2)
    return distance <= absorption_radius_pixels

# Main program
if __name__ == "__main__":
    async def setup_bluetooth():
        await bluetooth_manager.connect()

    async def cleanup_bluetooth():
        await bluetooth_manager.disconnect()

    asyncio.run(setup_bluetooth())  # Establish Bluetooth connection

    if not load_color_definitions():
        print("Color definitions not found. Entering color definition mode...")
        cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Unable to open the camera")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow('Define Colors - Click on the Camera Feed', frame)
            cv2.setMouseCallback('Define Colors - Click on the Camera Feed', pick_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        save_color_definitions()
        cap.release()
        cv2.destroyAllWindows()
    try:
        while True:
            # Input the target color
            target_color = input("Enter the target color: ").lower()
            print(f"Initial target color set to: {target_color}")

            if target_color not in color_definitions:
                print(f"Target color '{target_color}' is not defined. Exiting.")
                exit()

            cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                print("Unable to open the camera")
                exit()

            QR_CODE_SIZE = 0.067 # Actual size of the QR code (meters)
            ABSORPTION_RADIUS = 0.057 # Absorption radius (meters)
            RETURN_BASE_THRESHOLD = 0.057 + 0.07  # Return Base distance threshold in meters

            absorbed = False # Flag to check if the ball is absorbed
            returned_to_base = False  # Track if returned to base
            ABSORPTION_THRESHOLD = 5 # Number of consecutive frames to detect absorption
            absorption_count = 0



            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Print current target color
                print(f"Currently tracking target color: {target_color}")

                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame, qr_center, ppm, blue_midpoint, red_midpoint = detect_qr_code(frame, QR_CODE_SIZE)
                frame, target_ball_center, base_ball_center = detect_colors(frame, hsv_frame, target_color)

                if ppm is not None and ppm > 0 and base_ball_center is not None:
                    absorption_radius_pixels = ABSORPTION_RADIUS / ppm  # Convert absorption radius to pixels

                    if returned_to_base:
                        print("Already returned to Base.")
                        cv2.putText(frame, "Returned to Base", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        send_bluetooth_command("S 0")  # Send stop command
                        print("break!")
                        break
                    elif absorbed and base_ball_center is not None and qr_center is not None:
                        pixel_distance_to_base = math.sqrt(
                            (base_ball_center[0] - qr_center[0]) ** 2 + (base_ball_center[1] - qr_center[1]) ** 2
                        )
                        real_distance_to_base = pixel_distance_to_base * ppm

                        if real_distance_to_base <= RETURN_BASE_THRESHOLD:
                            returned_to_base = True  # Mark as returned to base
                            print("Successfully returned to Base!")
                            cv2.putText(frame, "Returned to Base", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            send_bluetooth_command("S 0")  # Send stop command
                            print("break!")
                            break
                        else:
                            rotation_command = calculate_rotation_angle(base_ball_center, blue_midpoint, red_midpoint)
                            if rotation_command != "Aligned":
                                print(f"QR code needs to rotate: {rotation_command}")
                                send_bluetooth_command(rotation_command)
                                cv2.putText(frame, f"Rotate: {rotation_command}",
                                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            else:
                                pixel_distance = math.sqrt(
                                    (base_ball_center[0] - qr_center[0]) ** 2 +
                                    (base_ball_center[1] - qr_center[1]) ** 2
                                )
                                real_distance = pixel_distance * ppm
                                movement_command = calculate_movement_command(real_distance)
                                print(f"Movement command: {movement_command}")
                                send_bluetooth_command(movement_command)
                                cv2.putText(frame, f"Move: {movement_command}",
                                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif not absorbed:
                        if target_ball_center is not None and blue_midpoint is not None:
                            if is_ball_absorbed(target_ball_center, blue_midpoint, ppm, absorption_radius_pixels):
                                absorption_count += 1
                                send_bluetooth_command("L 5")
                                if absorption_count >= ABSORPTION_THRESHOLD:
                                    absorbed = True
                                    print("Ball absorbed!")
                                    cv2.putText(frame, "Ball Absorbed", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                absorption_count = 0
                                rotation_command = calculate_rotation_angle(target_ball_center, blue_midpoint, red_midpoint)
                                if rotation_command != "Aligned":
                                    print(f"QR code needs to rotate: {rotation_command}")
                                    send_bluetooth_command(rotation_command)
                                    cv2.putText(frame, f"Rotate: {rotation_command}",
                                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                else:
                                    pixel_distance = math.sqrt(
                                        (target_ball_center[0] - qr_center[0]) ** 2 +
                                        (target_ball_center[1] - qr_center[1]) ** 2
                                    )
                                    real_distance = pixel_distance * ppm
                                    movement_command = calculate_movement_command(real_distance)
                                    print(f"Movement command: {movement_command}")
                                    send_bluetooth_command(movement_command)
                                    cv2.putText(frame, f"Move: {movement_command}",
                                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                
                
                cv2.imshow('Color and QR Code Detection', frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
    finally:
        asyncio.run(cleanup_bluetooth())
        cap.release()
        cv2.destroyAllWindows()
