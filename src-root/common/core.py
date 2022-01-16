import copy
import os
import re
import uuid

import yaml

ENV_VAR_PREFIX = "${"
ENV_VAR_SUFFIX = "}"
ENV_VAR_REGEX = r"^\${.+:?.+}$"


def config_value_parser(val):
    if isinstance(val, list):

        for i, e in enumerate(val):
            val[i] = config_value_parser(e)

    elif isinstance(val, dict):

        for k, v in val.items():
            val[k] = config_value_parser(v)

    elif isinstance(val, str):
        pattern = re.compile(ENV_VAR_REGEX)

        if pattern.fullmatch(val):
            env = val[len(ENV_VAR_PREFIX):-len(ENV_VAR_SUFFIX)].split(":", 1)
            val = os.getenv(*env)

    return val


class Context:
    __slots__ = ("uid", "conf", "params")

    def __init__(self, args):
        self.uid = uuid.uuid4().hex
        self.conf = Config(args)
        self.params = Parameter(args)


class Config:
    __slots__ = "resource"

    def __init__(self, args):
        with open(args.yml, mode="r") as yml_file:
            raw = yaml.load(yml_file, yaml.FullLoader) or {}
            raw = config_value_parser(raw)

            def protected():
                return copy.deepcopy(raw)

            self.resource = protected

    def get(self, path):
        conf = self.resource()

        flag = False
        ret = None
        try:
            for p in path.split("."):
                if flag is False:
                    ret = conf.pop(int(p)) if isinstance(conf, list) else conf.pop(p)
                    flag = True
                else:
                    ret = ret.pop(int(p)) if isinstance(ret, list) else ret.pop(p)
        except Exception:
            raise RuntimeError("Not found '%s' in config.", path)

        return ret

    def get_or_default(self, path, default):
        try:
            return self.get(path)
        except RuntimeError:
            return default


class Parameter:
    def __init__(self, args):
        params = {}

        for param in args.params:
            if "=" not in param:
                raise ValueError("Parameter must 'KEY=VALUE' string.")

            key, value = param.split("=")
            if key not in params:
                params[key] = value
            else:
                raise ValueError("Duplicate parameter: %s." % key)
        for k, v in params.items():
            setattr(self, k, v)
