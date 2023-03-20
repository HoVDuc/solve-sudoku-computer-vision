import yaml
import os

def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', 'yaml'], "Only support yaml file"
    stream = open(file_path, 'rb')
    config = yaml.load(stream, yaml.Loader)
    return config