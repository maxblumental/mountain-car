from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from .agent_params import AgentParams


@dataclass()
class PlayParams:
    steps_num: int
    agent_params: AgentParams


PlayParamsSchema = class_schema(PlayParams)


def read_play_params(path: str) -> PlayParams:
    with open(path, "r") as input_stream:
        schema = PlayParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
