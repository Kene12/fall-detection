import tkinter as tk
from tkinter import messagebox, Label, Entry, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import sys
import time
import requests
from threading import Thread, Event

# Set environment variable to avoid issues with duplicate libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Detection System")
        self.root.geometry("1280x720")  # Set window size to width 1280 height 720

        self.line_token = ""
        self.video_running = False
        self.thread = None
        self.stop_event = Event()
        self.sitting_detected_time = None
        self.fall_detected_time = None

        # Load the YOLOv8 model
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pt')
        self.model = YOLO(model_path)

        # Create menu
        menu = tk.Menu(root)
        root.config(menu=menu)

        camera_menu = tk.Menu(menu)
        menu.add_cascade(label="ดูกล้อง", menu=camera_menu)
        camera_menu.add_command(label="กล้อง 1", command=lambda: self.open_camera(0))
        camera_menu.add_command(label="กล้อง 2", command=lambda: self.open_camera(1))

        notify_menu = tk.Menu(menu)
        menu.add_cascade(label="แจ้งเตือน", menu=notify_menu)
        notify_menu.add_command(label="ทดสอบการแจ้งเตือน", command=self.test_notify)

        menu.add_command(label="Setting", command=self.open_settings)

        # Video display frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack(expand=True, fill=tk.BOTH)  # Adjust video frame size to fit the window

        # Settings frame (initially hidden)
        self.settings_frame = Frame(root)
        Label(self.settings_frame, text="กรอก Token Line:").pack(pady=10)
        self.token_entry = Entry(self.settings_frame, width=40)
        self.token_entry.pack(pady=10)
        Button(self.settings_frame, text="บันทึก", command=self.save_token).pack(pady=20)

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create images folder if it doesn't exist
        self.images_folder = os.path.join(os.path.dirname(__file__), 'images')
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)

    def open_camera(self, camera_id):
        self.hide_settings()
        if self.video_running:
            self.stop_camera()
        self.video_running = True
        self.stop_event.clear()
        if camera_id == 0:
            self.thread = Thread(target=self.run_video, args=(0,))
        else:
            self.thread = Thread(target=self.run_video, args=(os.path.join(os.path.dirname(__file__), 'test', 'Human Fall Detection Sample.mp4'),))
        self.thread.start()

    def run_video(self, source):
        # Path to video or camera
        if isinstance(source, str):
            video_path = source
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.video_running = False
                messagebox.showerror("Error", "ไม่พบกล้อง 1")
                return

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success:
                self.video_running = False
                cap.release()
                return

            # Run YOLOv8 detection
            results = self.model(frame)

            # Variables to keep track of the detected bounding boxes
            detected_boxes = []
            sitting_detected = False
            falling_detected = False

            # Set confidence threshold
            confidence_threshold = 0.5  # Adjust this value as needed

            # Extract person class detections and determine the highest priority detection
            for detection in results[0].boxes:
                if detection.conf < confidence_threshold:
                    continue

                class_id = int(detection.cls)
                x1, y1, x2, y2 = map(int, detection.xyxy[0])

                if class_id == 0:  # Class 'sitting'
                    color = (0, 255, 0)  # Green for sitting
                    label = "Sitting"
                    sitting_detected = True
                elif class_id == 1:  # Class 'standing'
                    color = (255, 0, 0)  # Blue for standing
                    label = "Standing"
                elif class_id == 2:  # Class 'falling'
                    color = (0, 0, 255)  # Red for falling
                    label = "Falling"
                    falling_detected = True
                else:
                    continue

                detected_boxes.append((x1, y1, x2, y2, color, label))

            # Check for sitting detection
            if sitting_detected:
                if self.sitting_detected_time is None:
                    self.sitting_detected_time = time.time()
                elif time.time() - self.sitting_detected_time >= 300:
                    self.send_line_notify("Alert: A person has been detected sitting for over 5 minutes!", self.line_token)
                    self.sitting_detected_time = None  # Reset after sending the notification
            else:
                self.sitting_detected_time = None

            # Check for falling detection
            if falling_detected:
                if self.fall_detected_time is None:
                    self.fall_detected_time = time.time()
                elif time.time() - self.fall_detected_time >= 60:
                    self.send_line_notify("Alert: A person has been detected falling for over 1 minute!", self.line_token)
                    # Save the frame
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = os.path.join(self.images_folder, f"fall_detected_{timestamp}.png")
                    cv2.imwrite(image_path, frame)
                    self.fall_detected_time = None  # Reset after sending the notification
            else:
                self.fall_detected_time = None

            # Draw the bounding boxes
            for (x1, y1, x2, y2, color, label) in detected_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert frame to ImageTk format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video frame in the main thread
            self.root.after(10, lambda: self.update_frame(imgtk))

        cap.release()
        self.video_running = False

    def update_frame(self, imgtk):
        if self.video_running:
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

    def send_line_notify(self, message, token):
        url = 'https://notify-api.line.me/api/notify'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': f'Bearer {token}'
        }
        payload = {'message': message}
        requests.post(url, headers=headers, data=payload)

    def test_notify(self):
        if self.line_token:
            self.send_line_notify("This is a test notification.", self.line_token)
            messagebox.showinfo("แจ้งเตือน", "การแจ้งเตือนทดสอบถูกส่งแล้ว")
        else:
            messagebox.showwarning("แจ้งเตือน", "กรุณากรอก Token Line ก่อน")

    def open_settings(self):
        self.hide_video()
        self.settings_frame.pack()

    def hide_settings(self):
        self.settings_frame.pack_forget()
        self.video_frame.pack()

    def hide_video(self):
        self.video_frame.pack_forget()

    def save_token(self):
        self.line_token = self.token_entry.get()
        messagebox.showinfo("Settings", "Token Line ถูกบันทึกแล้ว")
        self.hide_settings()

    def stop_camera(self):
        if self.video_running:
            self.video_running = False
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=1)
            self.stop_event.clear()

    def on_closing(self):
        self.stop_camera()
        self.root.quit()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()
    os._exit(0)
