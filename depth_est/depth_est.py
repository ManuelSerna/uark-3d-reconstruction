""" Depth estimation demo

NOTE: Using HugggingFace transformers V4.32.0
"""
import argparse
from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests


def get_base_parser():
    parser = argparse.ArgumentParser("Args Parser", add_help=False)

    # Cmd parameters
    parser.add_argument("--impath", type=str, required=True, help="input RGB image file path")
    parser.add_argument("--outpath", type=str, required=True, help="output path name to output depth image")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth Estimation Demo", parents=[get_base_parser()])
    args = parser.parse_args()
    #depth_filepath = os.path.join(args.depth)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(args.impath)

    image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save(args.outpath)
