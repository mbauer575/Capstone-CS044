# Maxwell Bauer
# 4/16/2025
# Label raw images

import argparse
import torch
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO on a folder of images."
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing raw images"
    )
    parser.add_argument(
        "output_folder",
        help="Path to the folder where labeled images will be saved"
    )
    args = parser.parse_args()

    print("Done loading, starting now.")
    print("PyTorch CUDA version:", torch.version.cuda)
    print("Beginning image annotations.")
    
    base_model = GroundingDINO(
        ontology=CaptionOntology({"toy car": "car"})
    )
    base_model.label(
        input_folder=args.input_folder,
        output_folder=args.output_folder
    )

if __name__ == "__main__":
    main()
