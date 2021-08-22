import logging

import click

from mountain_car.agent.q_learning import QLearningAgent
from mountain_car.hook import LoggerHook
from mountain_car.params import read_play_params, PlayParams
from mountain_car.play import play_mountain_car

logging.basicConfig(level=logging.DEBUG)


def train_agent(play_params: PlayParams):
    logging.info(f"start training agent with params: {play_params}")

    agent = QLearningAgent(play_params.agent_params)

    hooks = []
    if play_params.logger_params is not None:
        hooks.append(LoggerHook(play_params.logger_params))

    play_mountain_car(agent, hooks, play_params)


@click.command(name="train_agent")
@click.argument("config_path")
def train_agent_command(config_path: str):
    params = read_play_params(config_path)
    train_agent(params)


if __name__ == "__main__":
    train_agent_command()
