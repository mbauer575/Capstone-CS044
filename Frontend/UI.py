import os
import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from picamera2 import Picamera2
from picamera2.devices import Hailo
import serial.tools.list_ports

ports=serial.tools.list_ports.comports()
serialInst=serial.Serial()
serialInst.port="/dev/ttyACM0"
serialInst.baudrate=9600
serialInst.open()

# ---------------- Appearance ---------------- #
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ---------------- Model List (HEF paths) ---------------- #
model_paths = [
    "Models/cars.hef"
]
model_names = [os.path.basename(p) for p in model_paths]
model_map   = dict(zip(model_names, model_paths))

# ---------------- Globals ---------------- #
picam2 = None
hailo = None
class_names = []
video_w, video_h = 1280, 960
model_w = model_h = None
running = False

# ---------------- Helper Functions ---------------- #
def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    results = []
    for class_id, detections in enumerate(hailo_output):
        for det in detections:
            score = det[4]
            if score >= threshold:
                y0, x0, y1, x1 = det[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

# ---------------- Main Window ---------------- #
window = ctk.CTk()
window.title("Hailo YOLO UI on Pi")
window.geometry(f"{video_w+350}x{video_h+50}")
window.configure(fg_color="#fbf7ef")

object_count_var = tk.IntVar(value=0)
conf_var = tk.DoubleVar(value=0.0)
score_thresh = 0.5  # default threshold

# ---------------- Layout ---------------- #
frame = ctk.CTkFrame(window, fg_color="#fbf7ef")
frame.pack(fill="both", expand=True)

# Camera view
camera_label = ctk.CTkLabel(frame, text="", width=video_w, height=video_h)
camera_label.grid(row=0, column=0, padx=10, pady=10)

# Stats & Controls
control_panel = ctk.CTkFrame(frame, fg_color="#fbf7ef")
control_panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

ctk.CTkLabel(control_panel, text="Model:", font=("Helvetica", 14)).pack(pady=(0,5))
model_var = tk.StringVar(value=model_names[0])
ctk.CTkOptionMenu(control_panel, variable=model_var, values=model_names, width=200).pack()

start_btn = ctk.CTkButton(control_panel, text="Start", command=lambda: start_detection())
start_btn.pack(pady=(10,5))
stop_btn  = ctk.CTkButton(control_panel, text="Stop", command=lambda: stop_detection())
stop_btn.pack(pady=5)

ctk.CTkLabel(control_panel, text="Objects:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=object_count_var, font=("Helvetica", 24, "bold")).pack()

ctk.CTkLabel(control_panel, text="Avg Confidence:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=conf_var, font=("Helvetica", 20, "bold")).pack()

# ---------------- Detection Logic ---------------- #

def update_frame():
    global running, picam2, hailo, class_names
    if running and picam2:
        # Capture low-res for inference and full-res for display
        lores = picam2.capture_array('lores')
        main  = picam2.capture_array('main')

        # Run inference
        hailo_out = hailo.run(lores)
        detections = extract_detections(hailo_out, video_w, video_h, class_names, score_thresh)

        # Update stats
        count = len(detections)
        if count!=object_count_var:
            command=str(count)+"\n"
            serialInst.write(command.encode('utf-8'))
        object_count_var.set(count)
        avg_conf = sum([s for (_,_,s) in detections]) / count if count else 0.0
        conf_var.set(round(avg_conf, 2))

        # Draw boxes
        for (label, (x0, y0, x1, y1), score) in detections:
            cv2.rectangle(main, (x0, y0), (x1, y1), (0,255,0,0), 2)
            text = f"{label}:{score:.2f}"
            cv2.putText(main, text, (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0,0), 2)

        # Extract RGB and convert
        rgb = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        window.after(30, update_frame)


def start_detection():
    global picam2, hailo, class_names, model_w, model_h, video_w, video_h, running
    # Load Hailo model
    model_file = model_map[model_var.get()]
    hailo = Hailo(model_file)
    model_h, model_w, _ = hailo.get_input_shape()

    # Load labels
    label_file = model_file.replace('.hef', '.txt')
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()

    # Setup camera
    controls = {'FrameRate': 30}
    picam2 = Picamera2()
    main_cfg = {'size': (video_w, video_h), 'format': 'XRGB8888'}
    lores_cfg= {'size': (model_w, model_h), 'format': 'RGB888'}
    config = picam2.create_preview_configuration(main=main_cfg, lores=lores_cfg, controls=controls)
    picam2.configure(config)
    picam2.start()

    running = True
    update_frame()


def stop_detection():
    global running, picam2, hailo
    running = False
    object_count_var.set(0)
    conf_var.set(0.0)
    if picam2:
        picam2.stop()
        picam2.close()
        picam2 = None
    if hailo:
        hailo.close()
        hailo = None
    camera_label.configure(image=None)

# ---------------- Run ---------------- #
window.mainloop()
serialInst.close()