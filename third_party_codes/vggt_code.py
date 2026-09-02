"""Sky-segmentation helpers adapted from VGGT.

Source: https://github.com/facebookresearch/vggt
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is subject to the VGGT License:
# https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt

import copy
import os
import pathlib

import cv2
import numpy as np
import onnxruntime
import requests


def apply_sky_segmentation(image_files, target_size) -> np.ndarray:
    """Return non-sky masks resized to ``target_size`` for a batch of images."""
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            "skyseg.onnx",
        )

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        path = pathlib.Path(image_path)
        mask_filepath = path.parents[2] / "sky_masks" / path.parents[0].name / image_name.replace(".jpg", ".png")

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
            cv2.imwrite(mask_filepath, sky_mask)
        if sky_mask.shape[0] != target_size[0] or sky_mask.shape[1] != target_size[1]:
            sky_mask = cv2.resize(
                sky_mask,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    return (sky_mask_array > 0.1).astype(np.float32)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """Segment an image into sky (0) and non-sky (255) pixels."""
    assert mask_filename is not None
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """Run the sky-segmentation ONNX model for one BGR image."""
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    values = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    values = np.array(values, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    values = (values / 255 - mean) / std
    values = values.transpose(2, 0, 1)
    values = values.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = np.array(onnx_session.run([output_name], {input_name: values})).squeeze()
    min_value = np.min(result)
    max_value = np.max(result)
    result = (result - min_value) / (max_value - min_value)
    result *= 255
    return result.astype("uint8")


def download_file_from_url(url, filename):
    """Download a file while following the explicit Hugging Face redirect."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=8192):
                output_file.write(chunk)
        print(f"Downloaded {filename} successfully.")
    except requests.exceptions.RequestException as exc:
        print(f"Error downloading file: {exc}")
