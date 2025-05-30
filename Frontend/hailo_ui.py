#!/usr/bin/env python3
import os, json, time, argparse
import cv2
import numpy as np
from picamera2 import Picamera2, Preview, MappedArray
from picamera2.devices import Hailo
import serial

def extract_detections(hailo_out, w, h, names, thresh):
    results = []
    for cid, dets in enumerate(hailo_out):
        for d in dets:
            if d[4] >= thresh:
                y0,x0,y1,x1 = d[:4]
                bbox = (int(x0*w),int(y0*h),int(x1*w),int(y1*h))
                results.append((names[cid], bbox, d[4]))
    return results

def opencv_polygon_setup(json_path, video_w, video_h):
    print("== Interactive Polygon Setup ==")
    pts = []
    polys = []
    cap = cv2.VideoCapture(0)  # grab a frame from PiCam (fallback)
    _, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (video_w, video_h))
    temp = frame.copy()

    def draw_all(img):
        disp = img.copy()
        for poly in polys:
            cv2.polylines(disp, [np.array(poly)], True, (0,255,0),2)
        if pts:
            cv2.polylines(disp, [np.array(pts)], False, (0,0,255),1)
            for p in pts: cv2.circle(disp, tuple(p),5,(0,0,255),-1)
        return disp

    def callback(evt, x,y,flags,_) :
        nonlocal pts, polys, temp
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
        elif evt == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow("Setup")
    cv2.setMouseCallback("Setup", callback)
    print("Left click: add point  •  Right click: undo  •  c: close poly  •  s: save & exit  •  q: quit")
    while True:
        disp = draw_all(temp)
        cv2.imshow("Setup", disp)
        k = cv2.waitKey(1)&0xFF
        if k == ord('c') and len(pts)>=3:
            polys.append(pts.copy())
            pts.clear()
            print(f"Polygon #{len(polys)} added")
        elif k == ord('s'):
            with open(json_path,'w') as f:
                json.dump([ [ {'x':x,'y':y} for x,y in poly ] for poly in polys ], f, indent=2)
            print("Saved polygons.json")
            break
        elif k == ord('q'):
            print("Exited setup without saving")
            break
    cv2.destroyAllWindows()
    # convert back to list-of-list-of-dicts
    return [ [ {'x':x,'y':y} for x,y in poly ] for poly in polys ]

def load_polygons(path, video_w, video_h):
    if not os.path.exists(path):
        return opencv_polygon_setup(path, video_w, video_h)
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  help="HEF model path", default="Models/cars.hef")
    parser.add_argument("--labels",  help=".txt labels file", default="Models/cars.txt")
    parser.add_argument("--serial",  help="Serial port (e.g. /dev/ttyACM0)", default= "/dev/ttyACM0")
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args()

    # Serial init
    ser = serial.Serial(args.serial, 9600, timeout=1)

    # Load model & labels
    hailo = Hailo(args.model)
    m_h,m_w,_ = hailo.get_input_shape()
    with open(args.labels,'r') as f:
        class_names = f.read().splitlines()

    # Camera init
    video_w, video_h = 1280, 960
    picam = Picamera2()
    cfg = picam.create_preview_configuration(
        main={'size':(video_w,video_h),'format':'XRGB8888'},
        lores={'size':(m_w,m_h),'format':'RGB888'},
        controls={'FrameRate':30})
    picam.configure(cfg)
    picam.start_preview(Preview.QTGL)
    picam.start()

    # Load or setup polygons
    polygons = load_polygons("polygons.json", video_w, video_h)

    # Shared state
    detections = []
    last_send = {'cars':0,'spaces':len(polygons),'occ':0,'time':0}

    def draw_callback(request):
        nonlocal detections, last_send
        with MappedArray(request,"main") as m:
            frame = m.array
            # draw detections
            for name,(x0,y0,x1,y1),score in detections:
                cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)
                cv2.putText(frame,f"{name}:{int(score*100)}%",
                            (x0+5,y0+15),cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,0),1)
            # draw polygons & count occupied
            occ=0
            pts_np = lambda poly: np.array([[p['x'],p['y']] for p in poly],np.int32)
            for poly in polygons:
                col=(0,255,0)
                for _,(x0,y0,x1,y1),_ in detections:
                    cx,cy=( (x0+x1)//2,(y0+y1)//2 )
                    if cv2.pointPolygonTest(pts_np(poly),(cx,cy),False)>=0:
                        occ+=1
                        col=(0,0,255)
                        break
                cv2.polylines(frame,[pts_np(poly)],True,col,2)
            # overlay text metrics
            text_y = 30
            metrics = {
                'Cars': len(detections),
                'Spaces': len(polygons),
                'Occupied': occ,
                'AvgConf': round(sum(s for *_,s in detections)/len(detections) if detections else 0,2)
            }
            for k,v in metrics.items():
                cv2.putText(frame, f"{k}: {v}", (10, text_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255),2)
                text_y += 40

            # throttle serial updates to 1 Hz or on-change
            now = time.time()
            if now - last_send['time'] > 1:
                if (metrics['Cars']   != last_send['cars'] or
                    metrics['Spaces'] != last_send['spaces'] or
                    metrics['Occupied']!= last_send['occ']):
                    ser.write(f"{metrics['Spaces']}@".encode())
                    ser.write(f"{metrics['Occupied']}\n".encode())
                    last_send.update({
                        'cars':metrics['Cars'],
                        'spaces':metrics['Spaces'],
                        'occ':metrics['Occupied'],
                        'time': now})
    picam.pre_callback = draw_callback

    # Main loop: only low-res capture & inference
    try:
        while True:
            lo = picam.capture_array("lores")
            results = hailo.run(lo)
            detections = extract_detections(results, video_w, video_h, class_names, args.thresh)
    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        hailo.close()
        ser.close()

if __name__=="__main__":
    main()
