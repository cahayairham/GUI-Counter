import cv2
import tkinter as tk
from PIL import ImageTk, Image
import time
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create the main window
window = tk.Tk()
window.title("Webcam GUI")

# Create a label for displaying the webcam feed
label = tk.Label(window)
label.pack()

# Create a timer label
timer_label = tk.Label(window, text="00:00:00")
timer_label.pack()

# Create an entry field for inputting the timer time
timer_entry = tk.Entry(window)
timer_entry.pack()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables for timer
timer_running = False
start_time = 0
input_time = 0

# Function to capture and display webcam frames
def capture_frame():
    ret, frame = cap.read()
    
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with pose estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Run pose tracker
        result = pose.process(image=frame_rgb)
        pose_landmarks = result.pose_landmarks
        
        # Draw pose landmarks on the frame
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(frame_rgb, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
    
    # Convert frame back to BGR format
    frame_bgr = frame_rgb
    
    # Display the image in the GUI
    img = ImageTk.PhotoImage(image=Image.fromarray(frame_bgr))
    label.configure(image=img)
    label.image = img
    
    # Call the function again after a delay
    label.after(10, capture_frame)

# Function to start the timer
def start_timer():
    global timer_running, start_time, input_time
    if not timer_running:
        input_time = int(timer_entry.get())
        start_time = time.time()
        timer_running = True
        update_timer()

# Function to stop the timer
def stop_timer():
    global timer_running
    timer_running = False

# Function to reset the timer
def reset_timer():
    global start_time
    start_time = 0
    timer_label.config(text="00:00:00")

# Function to update the timer label
def update_timer():
    if timer_running:
        current_time = time.time()
        elapsed_seconds = int(current_time - start_time)
        remaining_seconds = input_time - elapsed_seconds
        if remaining_seconds <= 0:
            timer_label.config(text="00:00:00")
            stop_timer()
        else:
            minutes = remaining_seconds // 60
            seconds = remaining_seconds % 60
            hours = minutes // 60
            minutes = minutes % 60
            timer_label.config(text="{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))
            window.after(1000, update_timer)

# Create buttons for controlling the timer
start_button = tk.Button(window, text="Start Timer", command=start_timer)
start_button.pack()

stop_button = tk.Button(window, text="Stop Timer", command=stop_timer)
stop_button.pack()

# Start capturing and displaying webcam frames
capture_frame()

# Start the GUI event loop
window.mainloop()

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
