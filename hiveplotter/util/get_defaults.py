from configparser import ConfigParser
import os
import json
import site

CONF_PATH = os.path.join(site.getsitepackages()[0], 'hiveplotter_defaults.ini')


class Defaults():
    def __init__(self, config_path):
        cp = ConfigParser()
        cp.read(CONF_PATH)
        for section in cp.sections():
            for key, value in cp[section].items():
                self.__dict__[key] = json.loads(value)

        if config_path:
            cp.read(config_path)
            for section in cp.sections():
                for key, value in cp[section].items():
                    self.__dict__[key] = json.loads(value)