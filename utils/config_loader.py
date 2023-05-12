import pathlib
import sys

vis_rec_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(vis_rec_path.__str__())

from pathlib import Path
import os.path
import yaml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigLoader(metaclass=Singleton):
    def __init__(self):
        self.params = {}
        try:
            personal_config_file = Path(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets", "personal_config.yaml")))
            with open(personal_config_file, "r") as yaml_f:
                params = yaml.safe_load(yaml_f)
                if params is not None:
                    self.params = params
        except FileNotFoundError as ferr:
            print("Warning:", ferr)
            print(f"Creating an empty config file [{personal_config_file}] ... ")
            personal_config_file.touch(exist_ok=True)

    def get(self, param_name):
        if param_name in self.params.keys():
            return self.params[param_name]
        else:
            raise ValueError(f"Parameter {param_name} is not defined in the config file.")


Configs = ConfigLoader()

if __name__ == "__main__":
    config = ConfigLoader()
    img_data_dir = config.get("IMG_DATA_DIR")
    print(img_data_dir)
