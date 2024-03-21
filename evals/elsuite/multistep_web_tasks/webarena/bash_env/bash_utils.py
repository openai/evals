from dataclasses import dataclass

from evals.elsuite.multistep_web_tasks.webarena.core.env import EnvOutput, Observation


@dataclass
class BashObservation(Observation):
    output: str

    @property
    def data(self) -> str:
        return self.output


@dataclass
class BashEnvOutput(EnvOutput):
    observation: BashObservation
    reward: float
    done: bool
    truncated: bool = False
    info: None = None
