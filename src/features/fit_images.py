import argparse
import json
import os
import statistics
import sys
from time import time

import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
def remove_border(detections, np_image):
    height, width = np_image.shape[0], np_image.shape[1]

    if not detections:
        if height <= width:
            np_empty = np.zeros((height, height, 3), dtype="uint8")
            return np_empty
        else:
            np_empty = np.zeros((width, width, 3), dtype="uint8")
            return np_empty

    tops = []
    bottoms = []
    lefts = []
    rights = []

    for detection in detections:
        x1, y1, w_box, h_box = detection["bbox"]
        y_min, x_min, y_max, x_max = y1, x1, y1 + h_box, x1 + w_box

        # Convert to pixels, so we can use the PIL crop() function
        (left, right, top, bottom) = (
            x_min * width,
            x_max * width,
            y_min * height,
            y_max * height,
        )

        lefts.append(round(left))
        rights.append(round(right))

        tops.append(round(top))
        bottoms.append(round(bottom))

    lefts.sort()
    rights.sort(reverse=True)
    tops.sort()
    bottoms.sort(reverse=True)

    left = lefts[0]
    right = rights[0]
    top = tops[0]
    bottom = bottoms[0]

    new_height = bottom - top
    new_width = right - left

    np_cropped = np.zeros((new_height, new_width, 3), dtype="uint8")

    i_old = top
    i_new = 0

    j_old = left
    j_new = 0

    while i_old < bottom:
        while j_old < right:
            np_cropped[i_new][j_new] = np_image[i_old][j_old]
            j_old = j_old + 1
            j_new = j_new + 1
        i_old = i_old + 1
        i_new = i_new + 1

        j_old = lefts[0]
        j_new = 0

    return np_cropped


# ----------------------------------------------------------------------------------------------------------------------
def remove_borderV2(detections, np_image):
    height, width = np_image.shape[0], np_image.shape[1]

    if not detections:
        size = min(height, width)
        np_empty = np.zeros((size, size, 3), dtype="uint8")
        return np_empty

    bbox_list = [detection["bbox"] for detection in detections]
    bbox_array = np.array(bbox_list)

    x_min = bbox_array[:, 0]
    y_min = bbox_array[:, 1]
    w_box = bbox_array[:, 2]
    h_box = bbox_array[:, 3]

    x_max = x_min + w_box
    y_max = y_min + h_box

    left = int(x_min.min() * width)
    right = int(x_max.max() * width)
    top = int(y_min.min() * height)
    bottom = int(y_max.max() * height)

    new_height = bottom - top
    new_width = right - left

    np_cropped = np_image[top:bottom, left:right]

    return np_cropped


# ----------------------------------------------------------------------------------------------------------------------
def make_squared(np_image):
    height, width = np_image.shape[0], np_image.shape[1]
    im = Image.fromarray(np_image, "RGB")
    desired_size = 500
    if height < width:
        desired_size = width
    if height > width:
        desired_size = width
    if height == width:
        return im

    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is an in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )

    return new_im


# ----------------------------------------------------------------------------------------------------------------------
def make_squaredV2(np_image):
    height, width = np_image.shape[0], np_image.shape[1]
    desired_size = max(height, width)
    if height == width:
        return Image.fromarray(np_image, "RGB")

    old_size = (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # im = Image.fromarray(np_image, "RGB").resize(new_size, Image.ANTIALIAS)
    im = Image.fromarray(np_image, "RGB").resize(new_size, Image.Resampling.LANCZOS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )

    return new_im


# ----------------------------------------------------------------------------------------------------------------------
def run(data_images, output_masked, output_edited):
    assert len(data_images) > 0, "No input files provided"

    time_infer = []

    for sample in tqdm(data_images):
        image_path = sample["file"]

        name, ext = os.path.splitext(os.path.basename(image_path).lower())

        masked_path = os.path.join(output_masked, name + "_.png")
        edited_path = os.path.join(output_edited, name + "_.png")

        np_image = np.asarray(Image.open(masked_path), dtype="uint8")

        start_time = time()
        np_crop = remove_borderV2(sample["detections"], np_image)
        im = make_squaredV2(np_crop)
        elapsed_time = time() - start_time

        time_infer.append(elapsed_time)

        im.save(edited_path)

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = "NO DISPONIBLE"

    print("On average, for each image: ")
    print(
        "It took {} to adjust the image, with a deviation of {}".format(
            humanfriendly.format_timespan(average_time_infer), std_dev_time_infer
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Módulo para ajustar las imágenes enmascaradas convirtiéndolas en cuadradas"
    )
    parser.add_argument(
        "--json_file",
        help="Fichero JSON del que se tomarán los datos para delimitar la zona de la máscara",
    )
    parser.add_argument(
        "--output_masked",
        help="Ruta al directorio de donde se encuentran guardadas las imágenes enmascaradas",
    )
    parser.add_argument(
        "--output_edited",
        help="Ruta al directorio de donde se guardarán las imágenes ajustadas",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.json_file:
        json_file = args.json_file
    else:
        parser.print_help()
        parser.exit()

    if args.output_masked:
        os.makedirs(args.output_masked, exist_ok=True)
    else:
        parser.print_help()
        parser.exit()

    if args.output_edited:
        os.makedirs(args.output_edited, exist_ok=True)
    else:
        parser.print_help()
        parser.exit()

    with open(json_file, "r") as file:
        data = json.load(file)

    print("Ajustando {} imágenes...".format(len(data["images"])))

    run(data["images"], args.output_masked, args.output_edited)

    print("Resultados guardados en: {}".format(args.output_edited))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
