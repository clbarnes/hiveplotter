from configparser import ConfigParser
import os
import json

CONF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hiveplotter_defaults.ini')


class Defaults():
    def __init__(self, config_path):
        if config_path is None:
            config_path = CONF_PATH
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