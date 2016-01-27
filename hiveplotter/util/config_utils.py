from configparser import ConfigParser
import json
import os


CONF_PATH = os.path.join(__file__, '..', '..', 'hiveplotter_defaults.ini')


class Defaults:
    def __init__(self, config_path):
        cp = ConfigParser()
        cp.read(CONF_PATH)
        for section in cp.sections():
            for key, value in cp[section].items():
                setattr(self, key, json.loads(value))

        if config_path:
            cp.read(config_path)
            for section in cp.sections():
                for key, value in cp[section].items():
                    setattr(self, key, json.loads(value))


def ini_serialise(obj, path):
    configables = set()
    defaults = ConfigParser()
    defaults.read(CONF_PATH)

    for section in defaults.sections():
        configables.update(defaults[section])

    cfg_vals = dict()
    for key in dir(obj):
        if key not in configables:
            continue

        try:
            cfg_vals[key] = json.dumps(getattr(obj, key))
        except TypeError:
            pass

    categories = defaults.sections()
    category_dict = dict()
    for key in cfg_vals:
        if 'edge' in key:
            category_dict[key] = 'EDGE'
        elif 'background' in key:
            category_dict[key] = 'BACKGROUND'
        elif 'label' in key:
            category_dict[key] = 'AXIS_LABEL'
        elif 'node' in key:
            category_dict[key] = 'NODE'
        else:
            category_dict[key] = 'AXIS'

    cp = ConfigParser()

    for cat in sorted(categories):
        cp.add_section(cat)

    for key, value in sorted(cfg_vals.items()):
        cp.set(category_dict[key], key, value)

    with open(path, 'w') as f:
        cp.write(f)
