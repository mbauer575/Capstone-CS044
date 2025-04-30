# Maxwell Bauer
# 4/16/2024
# Train YOLO model

from ultralytics import YOLO
import torch
import argparse



def main():
    parser = argparse.ArgumentParser(
        description="Run Ultralytics on a custom dataset"
    )
    parser.add_argument(
        "output_folder",
        help="Path to the folder containing raw images"
    )
    parser.add_argument(
        "model_type",
        help="Type of YOLO model to train (yolov8n, yolov8s)"
    )
    parser.add_argument(
        "export_format",
        help="Export format for model (.pt or onnx)"
    )
    parser.add_argument(
        "img_size",
        help="Image size for training (e.g., 640, 1280)",
        type=int
    )
    args = parser.parse_args()

    print("Done loading, starting now.")
    print("PyTorch CUDA version:", torch.version.cuda)
    print("Beginning training.")
    
    model = YOLO("yolov8n.pt")  # Load yolov8n.pt
    # Train the model
    results = model.train(data=args.output_folder+'/data.yaml', epochs=200, imgsz=args.img_size, batch=16, device=[0])

    model.export(format=args.export_format)

    model = YOLO("yolov8s.pt")  # Load yolov8s.pt
    # Train the model
    results = model.train(data=args.output_folder+'/data.yaml', epochs=200, imgsz=args.img_size, batch=16, device=[0])

    model.export(format=args.export_format)


if __name__ == "__main__":
    main()