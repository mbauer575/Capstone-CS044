import os
import customtkinter as ctk
import tkinter as tk
import cv2
import json
import numpy as np
from PIL import Image, ImageTk
from picamera2 import Picamera2
from picamera2.devices import Hailo
import serial.tools.list_ports

# ---------------- Serial Init ---------------- #
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
serialInst.port = "/dev/ttyACM0"
serialInst.baudrate = 9600
serialInst.open()

# ---------------- Appearance ---------------- #
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ---------------- Model List ---------------- #
model_paths = ["Models/cars.hef"]
model_names = [os.path.basename(p) for p in model_paths]
model_map = dict(zip(model_names, model_paths))

# ---------------- Globals ---------------- #
picam2 = None
hailo = None
class_names = []
video_w, video_h = 1280, 960
model_w = model_h = None
running = False
polygons = []  # Loaded polygons from JSON

# ---------------- Helper ---------------- #
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

def point_in_polygon(point, polygon):
    pts = np.array([[p["x"], p["y"]] for p in polygon], np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0

# ---------------- Main Window ---------------- #
window = ctk.CTk()
window.title("Hailo YOLO UI on Pi")
window.geometry(f"{video_w+350}x{video_h+50}")
window.configure(fg_color="#fbf7ef")

object_count_var = tk.IntVar(value=0)
conf_var = tk.DoubleVar(value=0.0)
score_thresh = 0.5
num_parking_spaces = tk.IntVar(value=0)
num_occupied_spaces = tk.IntVar(value=0)

# ---------------- Layout ---------------- #
frame = ctk.CTkFrame(window, fg_color="#fbf7ef")
frame.pack(fill="both", expand=True)

camera_label = ctk.CTkLabel(frame, text="", width=video_w, height=video_h)
camera_label.grid(row=0, column=0, padx=10, pady=10)

control_panel = ctk.CTkFrame(frame, fg_color="#fbf7ef")
control_panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

ctk.CTkLabel(control_panel, text="Model:", font=("Helvetica", 14)).pack(pady=(0,5))
model_var = tk.StringVar(value=model_names[0])
ctk.CTkOptionMenu(control_panel, variable=model_var, values=model_names, width=200).pack()

ctk.CTkButton(control_panel, text="Start", command=lambda: start_detection()).pack(pady=(10,5))
ctk.CTkButton(control_panel, text="Stop", command=lambda: stop_detection()).pack(pady=5)
ctk.CTkButton(control_panel, text="Set Up", command=lambda: set_markers()).pack(pady=5)

ctk.CTkLabel(control_panel, text="Cars:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=object_count_var, font=("Helvetica", 24, "bold")).pack()

ctk.CTkLabel(control_panel, text="Parking Spaces:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=num_parking_spaces, font=("Helvetica", 24, "bold")).pack()

ctk.CTkLabel(control_panel, text="Occupied Spaces:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=num_occupied_spaces, font=("Helvetica", 24, "bold")).pack()

ctk.CTkLabel(control_panel, text="Avg Confidence:", font=("Helvetica", 14)).pack(pady=(20,0))
ctk.CTkLabel(control_panel, textvariable=conf_var, font=("Helvetica", 20, "bold")).pack()

# ---------------- Detection Logic ---------------- #
def update_frame():
    global running, picam2, hailo, class_names, polygons
    if running and picam2:
        lores = picam2.capture_array('lores')
        main  = picam2.capture_array('main')

        hailo_out = hailo.run(lores)
        detections = extract_detections(hailo_out, video_w, video_h, class_names, score_thresh)

        count = len(detections)
        if count != object_count_var.get():
            command = str(count) + "\n"
            serialInst.write(command.encode('utf-8'))
        object_count_var.set(count)
        avg_conf = sum([s for (_,_,s) in detections]) / count if count else 0.0
        conf_var.set(round(avg_conf, 2))


        # Count parking spaces
        num_parking_spaces.set(len(polygons))

        # Reset the occupied spaces counter at the start of each frame
        occupied_count = 0

        # List to track which polygons are occupied
        occupied_polygons = set()

        # Draw polygons: green by default, red if any detected object center inside
        for poly in polygons:
            color = (0, 255, 0)  # green by default
            for (_, (x0, y0, x1, y1), _) in detections:
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                if point_in_polygon((cx, cy), poly):
                    # Increment occupied spaces count only if this polygon is not already marked
                    if id(poly) not in occupied_polygons:
                        occupied_count += 1
                        occupied_polygons.add(id(poly))  # Mark this polygon as occupied
                    color = (0, 0, 255)  # red if object detected inside the polygon
                    break
            pts = np.array([[p["x"], p["y"]] for p in poly], np.int32)
            cv2.polylines(main, [pts], isClosed=True, color=color, thickness=2)

        # Update the UI with the correct occupied count
        num_occupied_spaces.set(occupied_count)

        # Draw detections
        for (label, (x0, y0, x1, y1), score) in detections:
            cv2.rectangle(main, (x0, y0), (x1, y1), (0,255,0), 2)
            text = f"{label}:{score:.2f}"
            cv2.putText(main, text, (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        rgb = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        window.after(30, update_frame)

def load_polygons():
    global polygons
    try:
        with open("polygons.json", "r") as f:
            polygons = json.load(f)
        print(f"Loaded {len(polygons)} polygons from polygons.json")
    except FileNotFoundError:
        print("No polygons.json file found.")
        polygons = []

def start_detection():
    global picam2, hailo, class_names, model_w, model_h, video_w, video_h, running
    model_file = model_map[model_var.get()]
    hailo = Hailo(model_file)
    model_h, model_w, _ = hailo.get_input_shape()

    label_file = model_file.replace('.hef', '.txt')
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()

    controls = {'FrameRate': 30}
    picam2 = Picamera2()
    main_cfg = {'size': (video_w, video_h), 'format': 'XRGB8888'}
    lores_cfg= {'size': (model_w, model_h), 'format': 'RGB888'}
    config = picam2.create_preview_configuration(main=main_cfg, lores=lores_cfg, controls=controls)
    picam2.configure(config)
    picam2.start()

    load_polygons()  # Load polygons on start

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

def set_markers():
    if not picam2:
        print("Camera not started.")
        return

    frame = picam2.capture_array("main")

    scale = 0.5
    snapshot = frame.copy()
    snapshot_display = cv2.resize(snapshot, (0, 0), fx=scale, fy=scale)
    local_polygons = polygons.copy()  # Work on a local copy to avoid partial update if user cancels
    current_polygon = []
    delete_mode = False

    def update_display():
        display = cv2.resize(snapshot.copy(), (0, 0), fx=scale, fy=scale)
        for poly in local_polygons:
            pts = np.array([[int(p["x"] * scale), int(p["y"] * scale)] for p in poly], np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if current_polygon:
            pts = np.array([[int(p["x"] * scale), int(p["y"] * scale)] for p in current_polygon], np.int32)
            cv2.polylines(display, [pts], isClosed=False, color=(0, 0, 255), thickness=1)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
        return display

    def point_in_polygon(point, polygon):
        pts = np.array([[p["x"], p["y"]] for p in polygon], np.int32)
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon, snapshot_display, delete_mode

        x_orig = int(x / scale)
        y_orig = int(y / scale)

        if delete_mode and event == cv2.EVENT_LBUTTONDOWN:
            for i, poly in enumerate(local_polygons):
                if point_in_polygon((x_orig, y_orig), poly):
                    print(f"Deleted polygon #{i}")
                    del local_polygons[i]
                    snapshot_display = update_display()
                    return

        if not delete_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                current_polygon.append({"x": x_orig, "y": y_orig})
                snapshot_display = update_display()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if current_polygon:
                    current_polygon.pop()
                    snapshot_display = update_display()

    cv2.namedWindow("Set Polygons")
    cv2.setMouseCallback("Set Polygons", mouse_callback)

    print("Polygon Tool:\n"
          "- Left-click to add points\n"
          "- Right-click to undo\n"
          "- Press 'c' to close polygon\n"
          "- Press 'd' to toggle delete mode\n"
          "- Press 's' to save and exit\n"
          "- Press 'q' to quit without saving")

    while True:
        cv2.imshow("Set Polygons", snapshot_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if len(current_polygon) >= 3:
                local_polygons.append(current_polygon.copy())
                current_polygon.clear()
                snapshot_display = update_display()
                print("Polygon added.")
            else:
                print("Need at least 3 points.")
        elif key == ord("d"):
            delete_mode = not delete_mode
            print("Delete mode:", "ON" if delete_mode else "OFF")
        elif key == ord("s"):
            with open("polygons.json", "w") as f:
                json.dump(local_polygons, f, indent=4)
            print("Saved to polygons.json")
            break
        elif key == ord("q"):
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()

    # Update global polygons after editing
    polygons = local_polygons.copy()

    #Send to Arduino
    command = str(polygons) + "@"
    serialInst.write(command.encode('utf-8'))

# ---------------- Run ---------------- #
window.mainloop()
serialInst.close()
