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