# ---------------- Libraries  ---------------- #
import os
import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import serial.tools.list_ports
from picamera2.devices import Hailo

# ---------------- Setup Arduino ------------ #
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

for port in ports:
    if "Arduino" in str(port):
        serialInst.port = str(port)[:4]

serialInst.baudrate = 9600
if serialInst.port != None:
    serialInst.open()

# ---------------- Appearance ---------------- #
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ---------------- Model List (full paths) ---------------- #
model_paths = [
    "Models/yolov8s.pt",
    "Models/yolov8n.pt",
    "Models/yolo11n.pt",
    "Models/ToyCarsV1.pt"
]

# Build display names + lookup map
model_names = [os.path.basename(p) for p in model_paths]
model_map   = dict(zip(model_names, model_paths))
# ---------------- Globals ---------------- #
model_h, model_w, _ = hailo.get_input_shape()
video_w, video_h = 1280, 960

with open('cars.idunno', 'r', encoding="utf-8") as f:
    class_names = f.read().splitlines()
cap = None
model = None
running = False
lores = {'size': (model_w, model_h), 'format': 'RGB888'}

# ---------------- Main Window ---------------- #
window = ctk.CTk()
window.title("YOLO Detection UI")
window.geometry("1000x600")
window.configure(fg_color="#fbf7ef")

object_count_var = tk.IntVar(value=0)
conf_var = tk.DoubleVar(value=0.0)
# ---------------- Tabview ---------------- #
tabs = ctk.CTkTabview(window, width=900, height=550)
tabs.pack(padx=20, pady=20, fill="both", expand=True)
tabs.add("Live View")
tabs.add("Settings")

# ---------------- Live View Tab ---------------- #
live_tab = tabs.tab("Live View")
live_tab.configure(fg_color="#fbf7ef")
live_tab.grid_rowconfigure(0, weight=1)
live_tab.grid_columnconfigure(0, weight=1)
live_tab.grid_columnconfigure(1, weight=0)

# Camera Feed Frame
camera_frame = ctk.CTkFrame(live_tab, width=500, height=500, fg_color="#7e7373", corner_radius=10)
camera_frame.grid(row=0, column=0, padx=20, pady=20)
camera_label = ctk.CTkLabel(camera_frame, text="", width=500, height=500, text_color="white")
camera_label.pack()

# Stats Panel
right_panel = ctk.CTkFrame(live_tab, width=200, height=500, fg_color="#fbf7ef", corner_radius=10)
right_panel.grid(row=0, column=1, padx=20, pady=20)

# Object Count
ctk.CTkLabel(
    right_panel, text="Objects Detected",
    font=("Helvetica", 16, "bold"),
    text_color="black"
).pack(pady=(20, 5))
ctk.CTkLabel(
    right_panel, textvariable=object_count_var,
    font=("Helvetica", 36, "bold"),
    text_color="#4a4a4a"
).pack()

# Average Confidence
ctk.CTkLabel(
    right_panel, text="Avg. Confidence",
    font=("Helvetica", 14, "bold"),
    text_color="black"
).pack(pady=(20, 5))
ctk.CTkLabel(
    right_panel, textvariable=conf_var,
    font=("Helvetica", 24, "bold"),
    text_color="#4a4a4a"
).pack()

# ---------------- Settings Tab ---------------- #
settings_tab = tabs.tab("Settings")
settings_tab.configure(fg_color="#fbf7ef")

control_frame = ctk.CTkFrame(settings_tab, fg_color="#fbf7ef")
control_frame.pack(pady=40)

# Model Selector
model_var = tk.StringVar(value=model_names[0])
ctk.CTkOptionMenu(
    control_frame,
    variable=model_var,
    values=model_names,
    text_color="black",
    width=200, height=40,
    corner_radius=20,
    font=("Helvetica", 13),
    dropdown_font=("Helvetica", 12)
).pack(side=tk.LEFT, padx=20)

# Start Button
ctk.CTkButton(
    control_frame,
    text="Start",
    command=lambda: start_detection(),
    fg_color="#fcd34d",
    text_color="black",
    hover_color="#ffc107",
    width=120, height=40,
    corner_radius=20,
    font=("Helvetica", 13)
).pack(side=tk.LEFT, padx=20)

# Stop Button
ctk.CTkButton(
    control_frame,
    text="Stop",
    command=lambda: stop_detection(),
    fg_color="#ff4c4c",
    text_color="black",
    hover_color="#ff1a1a",
    width=120, height=40,
    corner_radius=20,
    font=("Helvetica", 13)
).pack(side=tk.LEFT, padx=20)

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

# ---------------- Detection Logic ---------------- #
def update_frame():
    global cap, model, running
    if running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            window.after(10, update_frame)
            return

        # Run YOLO inference
        # results = model(frame)[0]
        # detections = results.boxes

        # Run YOLO model with Hailo
        results = model.run(frame)
        detections = extract_detections(results, video_w, video_h, class_names, 0.7)

        # If the amount of detections has changed we need to update the arduino
        count = len(detections)
        if serialInst.isOpen() and count != object_count_var:
            command=str(count)+"\n"
            serialInst.write(command.encode('utf-8'))
        object_count_var.set(count)

        # Compute average confidence
        confidences = [float(box.conf[0]) for box in detections]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        conf_var.set(round(avg_conf, 2))

        # Draw bounding boxes and labels
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])  # Extract confidence

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Annotate confidence score next to object label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            
        # Convert to RGB and resize to fit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).resize((750, 750))
        imgtk = ImageTk.PhotoImage(img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        window.after(10, update_frame)

def start_detection():
    global cap, model, running
    selected_name = model_var.get()       
    model_path    = model_map[selected_name]
    model         = Hailo(model_path)
    cap           = cv2.VideoCapture(0)
    running       = True
    update_frame()


def stop_detection():
    global running, cap
    running = False
    object_count_var.set(0)                # reset stats
    conf_var.set(0.0)
    if cap:                               # release and clear camera
        cap.release()
        cap = None
    camera_label.imgtk = None            # clear the image from the label
    camera_label.configure(image=None)

# ---------------- Run ---------------- #
window.mainloop()

# ---------- Arduino Cleanup -----------#
serialInst.close()