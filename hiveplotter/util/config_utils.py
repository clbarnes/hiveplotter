from configparser import ConfigParser
import json
import os
import logging
import sys
from io import StringIO

logger = logging.getLogger()

CONF_PATH = os.path.abspath(os.path.join(__file__, '..', '..', 'hiveplotter_defaults.ini'))
logger.debug('Config path: {}'.format(CONF_PATH))


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

    def __str__(self):
        return str({
            key: getattr(self, key) for key in dir(self) if not key.startswith('_')
        })


def ini_serialise(obj, path=None):
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
            cfg_vals[key] = json.dumps(getattr(obj, key), sort_keys=True)
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

    if path is None:
        out = StringIO()
        cp.write(out)
        return out.getvalue()
    else:
        with open(path, 'w') as f:
            cp.write(f)
