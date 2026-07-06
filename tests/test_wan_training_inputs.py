import importlib.util
from pathlib import Path


def load_train_module():
    module_path = Path("examples/wanvideo/model_training/train.py").resolve()
    spec = importlib.util.spec_from_file_location("wan_train_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_image_extra_input_prefers_loaded_metadata_column():
    train_module = load_train_module()
    module = train_module.WanTrainingModule.__new__(train_module.WanTrainingModule)

    inputs = module.parse_extra_inputs(
        {"video": ["video_frame_0"], "input_image": ["condition_image"]},
        ["input_image"],
        {},
    )

    assert inputs["input_image"] == "condition_image"


def test_input_image_extra_input_falls_back_to_video_first_frame():
    train_module = load_train_module()
    module = train_module.WanTrainingModule.__new__(train_module.WanTrainingModule)

    inputs = module.parse_extra_inputs({"video": ["video_frame_0"]}, ["input_image"], {})

    assert inputs["input_image"] == "video_frame_0"
