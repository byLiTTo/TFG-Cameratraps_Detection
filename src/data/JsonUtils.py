import datetime
import json
import os
import platform

from models.Detection import TFDetector


# ----------------------------------------------------------------------------------------------------------------------
def generate_json(results, output_dir):
    """
    Generate a JSON file containing the detection results.

    Args:
        results (list): A list of dictionaries representing the detection results for each image.
        output_dir (str): The directory where the JSON file will be saved.

    Returns:
        None
    """
    try:
        final_output = {
            "images": results,
            "detection_categories": TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
            "info": {
                "detection_completion_time": datetime.datetime.utcnow().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "format_version": "1.0",
            },
        }

        name = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
        file_name = name + ".json"

        os.makedirs((output_dir + os.sep + "registry"), exist_ok=True)
        output_file = output_dir + os.sep + "registry" + os.sep + file_name

        with open(output_file, "w") as f:
            json.dump(final_output, f, indent=1)

        for i_image in final_output["images"]:
            fn = i_image["file"]
            file_name = os.path.basename(fn).lower()
            name, ext = os.path.splitext(file_name)
            i_results = {
                "file": fn,
                "max_detection_conf": i_image["max_detection_conf"],
                "detections": i_image["detections"],
                "detection_categories": TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
            }
            output_file = output_dir + os.sep + name + ".json"

            with open(output_file, "w") as f:
                json.dump(i_results, f, indent=1)

    except Exception as e:
        print("An error occurred:", str(e))
