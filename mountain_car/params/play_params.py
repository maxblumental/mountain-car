from dataclasses import dataclass
from typing import Optional

import yaml
from marshmallow_dataclass import class_schema

from .agent_params import AgentParams
from .logger_params import LoggerParams


@dataclass()
class PlayParams:
    steps_num: int
    agent_params: AgentParams
    logger_params: Optional[LoggerParams]
    dqn_path: str


PlayParamsSchema = class_schema(PlayParams)


def read_play_params(path: str) -> PlayParams:
    with open(path, "r") as input_stream:
        schema = PlayParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
