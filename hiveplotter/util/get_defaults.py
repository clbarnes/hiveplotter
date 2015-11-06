from configparser import ConfigParser
import json
from pkg_resources import resource_string

CONF_PATH = resource_string('hiveplotter', 'hiveplotter_defaults.ini')


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