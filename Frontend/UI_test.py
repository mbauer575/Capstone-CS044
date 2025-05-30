#!/usr/bin/env python3

import os
import sys
import json
import cv2
import numpy as np
import customtkinter as ctk
import tkinter as tk
from PIL import Image
from customtkinter import CTkImage
from picamera2 import Picamera2
from picamera2.devices import Hailo
import serial

# ---------------- Appearance & Root Window ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Hailo YOLO UI on Pi")
video_w, video_h = 1280, 960
window.geometry(f"{video_w+350}x{video_h+50}")
window.configure(fg_color="#fbf7ef")

# ---------------- Serial Init (optional) ----------------
try:
    serialInst = serial.Serial(port="/dev/ttyACM0", baudrate=9600, timeout=1)
except Exception as e:
    print(f"Warning: serial display not found — continuing without it: {e}")
    serialInst = None

# ---------------- Tk Variables ----------------
object_count_var    = tk.IntVar(window, value=0)
conf_var            = tk.DoubleVar(window, value=0.0)
num_parking_spaces  = tk.IntVar(window, value=0)
num_occupied_spaces = tk.IntVar(window, value=0)

# ---------------- Model Config ----------------
model_paths = ["Models/cars.hef", "Models/yolov8s.hef"]
model_names = [os.path.basename(p) for p in model_paths]
model_map   = dict(zip(model_names, model_paths))
score_thresh = 0.5

# ---------------- Globals ----------------
picam2      = None
hailo       = None
class_names = []
polygons    = []
running     = False
last_future = None

# ---------------- Helper Functions ----------------
def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    results = []
    for cid, dets in enumerate(hailo_output):
        for d in dets:
            if (score := d[4]) >= threshold:
                y0, x0, y1, x1 = d[:4]
                bbox = (int(x0*w), int(y0*h), int(x1*w), int(y1*h))
                results.append((class_names[cid], bbox, score))
    return results

def point_in_poly(pt, poly):
    pts = np.array([[p["x"], p["y"]] for p in poly], np.int32)
    return cv2.pointPolygonTest(pts, pt, False) >= 0

def load_polygons():
    global polygons
    try:
        with open("polygons.json") as f:
            polygons = json.load(f)
        print(f"Loaded {len(polygons)} polygons")
    except FileNotFoundError:
        print("No polygons.json file found.")
        polygons = []

# ---------------- Detection Loop ----------------
def update_frame():
    global last_future

    if not running or not picam2:
        return

    # 1) grab both streams
    with picam2.captured_request() as req:
        lores = req.make_array("lores")
        main  = req.make_array("main")

    # 2) fire off async inference
    current_future = hailo.run_async(lores)

    # 3) handle results from previous frame
    if last_future and last_future.done():
        outputs = last_future.result()
        dets = extract_detections(outputs, video_w, video_h, class_names, score_thresh)

        # Update counts
        object_count_var.set(len(dets))
        avg_c = sum(s for _,_,s in dets)/len(dets) if dets else 0.0
        conf_var.set(round(avg_c,2))

        # Parking occupancy
        occupied = 0
        seen_ids = set()
        for poly in polygons:
            for _,(x0,y0,x1,y1),_ in dets:
                cx, cy = (x0+x1)//2, (y0+y1)//2
                if point_in_poly((cx,cy), poly) and id(poly) not in seen_ids:
                    seen_ids.add(id(poly))
                    occupied += 1
                    break

        num_parking_spaces.set(len(polygons))
        num_occupied_spaces.set(occupied)

        # Serial display
        if serialInst:
            serialInst.write(f"{len(dets)}\n".encode())
            serialInst.write(f"{len(polygons)}@\n".encode())
            serialInst.write(f"{occupied}\n".encode())

        # Draw polygons & detections
        for poly in polygons:
            pts = np.array([[p["x"],p["y"]] for p in poly], np.int32)
            color = (0,0,255) if id(poly) in seen_ids else (0,255,0)
            cv2.polylines(main, [pts], True, color, 2)

        for lbl,(x0,y0,x1,y1),s in dets:
            cv2.rectangle(main, (x0,y0),(x1,y1), (0,255,0), 2)
            cv2.putText(main, f"{lbl}:{s:.2f}", (x0+5,y0+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    last_future = current_future

    # 4) render to CTkImage
    rgb    = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb)
    ctk_im = CTkImage(light_image=pil_im, size=(video_w, video_h))
    camera_label.configure(image=ctk_im)
    camera_label.image = ctk_im

    # 5) schedule next
    window.after_idle(update_frame)

# ---------------- Control Functions ----------------
def start_detection():
    global picam2, hailo, class_names, running

    if running:
        stop_detection()

    # Load Hailo model
    hef = model_map[model_var.get()]
    hailo = Hailo(hef)
    mh, mw, _ = hailo.get_input_shape()
    with open(hef.replace('.hef','.txt')) as f:
        class_names = f.read().splitlines()

    # Configure PiCamera2
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={'size':(video_w,video_h),'format':'XRGB8888'},
        lores={'size':(mw,mh),'format':'RGB888'},
        controls={'FrameRate':30})
    picam2.configure(cfg)
    picam2.start()

    load_polygons()
    running = True
    update_frame()

def stop_detection():
    global running, picam2, hailo
    running = False
    object_count_var.set(0)
    conf_var.set(0.0)
    if picam2:
        picam2.stop(); picam2.close(); picam2 = None
    if hailo:
        hailo.close(); hailo = None
    camera_label.configure(image=None)

def restart_app():
    stop_detection()
    os.execv(sys.executable, [sys.executable] + sys.argv)

def set_markers():
    global polygons
    if not picam2:
        print("Camera not started.")
        return
    # … your polygon-annotation code here …

# ---------------- UI Construction ----------------
frame = ctk.CTkFrame(window, fg_color="#fbf7ef")
frame.pack(fill="both", expand=True)

camera_label = ctk.CTkLabel(frame, width=video_w, height=video_h)
camera_label.grid(row=0, column=0, padx=10, pady=10)

control_panel = ctk.CTkFrame(frame, fg_color="#fbf7ef")
control_panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

# Model selector + buttons
ctk.CTkLabel(control_panel, text="Model:", font=("Helvetica",14)).pack(pady=(0,5))
model_var = tk.StringVar(window, value=model_names[0])
ctk.CTkOptionMenu(control_panel, variable=model_var, values=model_names, width=200).pack()

ctk.CTkButton(control_panel, text="Start",   command=start_detection).pack(pady=(10,5))
ctk.CTkButton(control_panel, text="Restart", command=restart_app).pack(pady=5)
ctk.CTkButton(control_panel, text="Set Up",  command=set_markers).pack(pady=5)

# Counters
for label, var in [
    ("Cars:", object_count_var),
    ("Parking Spaces:", num_parking_spaces),
    ("Occupied Spaces:", num_occupied_spaces),
    ("Avg Confidence:", conf_var),
]:
    ctk.CTkLabel(control_panel, text=label, font=("Helvetica",14)).pack(pady=(20,0))
    ctk.CTkLabel(control_panel, textvariable=var, font=("Helvetica",24,"bold")).pack()

# ---------------- Run ----------------
window.mainloop()

if serialInst and serialInst.is_open:
    serialInst.close()
