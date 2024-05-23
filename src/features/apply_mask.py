import argparse
import json
import os
import statistics
import sys
import traceback
from time import time

import cv2
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
def generate_masked_image(image_path, mask_path):
    try:
        image = cv2.imread(image_path)
        bin_mask = np.array(Image.open(mask_path))

        masked = cv2.bitwise_and(image, image, mask=bin_mask)

        return masked
    except Exception as e:
        print("An error occurred while generating the masked image:")
        print(e)
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------------------------------------------------
def run(file_names, output_mask, output_masked):
    assert len(file_names) > 0, "No input files provided"

    time_infer = []

    for sample in tqdm(file_names):
        try:
            image_path = sample["file"]

            name, ext = os.path.splitext(os.path.basename(image_path).lower())

            mask_image = os.path.join(output_mask, name + "_mask.png")
            masked_file_name = os.path.join(output_masked, name + "_.png")

            # image = np.array(Image.open(image_path))
            start_time = time()
            masked = generate_masked_image(image_path, mask_image)
            elapsed_time = time() - start_time

            time_infer.append(elapsed_time)

            cv2.imwrite(masked_file_name, masked)
        except Exception as e:
            print("An error occurred while processing image: {}".format(image_path))
            print(e)
            traceback.print_exc()

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = "NO DISPONIBLE"

    print("On average, for each image: ")
    print(
        "It took {} to apply the mask, with a deviation of {}".format(
            humanfriendly.format_timespan(average_time_infer), std_dev_time_infer
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Módulo para aplicar las máscaras anteriormente generadas a las imágenes indicadas"
    )

    parser.add_argument(
        "--json_file",
        help="Fichero JSON del que se tomará la ruta de la imagen original a la que aplicar la máscara",
    )
    parser.add_argument(
        "--output_mask",
        help="Ruta al directorio de donde se encuentran los ficheros de las máscaras ",
    )
    parser.add_argument(
        "--output_masked",
        help="Ruta al directorio de donde se guardaran las imágenes una vez aplicadas sus máscaras"
        "correspondientes",
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

    if args.output_mask:
        os.makedirs(args.output_mask, exist_ok=True)
    else:
        parser.print_help()
        parser.exit()

    if args.output_masked:
        os.makedirs(args.output_masked, exist_ok=True)
    else:
        parser.print_help()
        parser.exit()

    with open(json_file, "r") as file:
        data = json.load(file)

    print("Applying masks to {} images...".format(len(data["images"])))

    run(data["images"], args.output_mask, args.output_masked)

    print("Results saved in: {}".format(args.output_masked))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
