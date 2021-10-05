import yaml


def read_yaml_file(filename) -> dict:
    with open(f"{filename}") as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    return d
