import os
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
model_file = "D:/mliot/Mateusz_1992-objectdetectionandroid-5c2211ab9096/object_detection/android/app/src/main/assets/detect.tflite"
displayer = _metadata.MetadataDisplayer.with_model_file(model_file)
export_json_file = os.path.join(os.path.splitext(model_file)[0] + ".json")
json_file = displayer.get_metadata_json()
with open(export_json_file, "w") as f:
    f.write(json_file)