from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Optional, TypedDict


@dataclass
class EarlyStopConfig:
    max_steps: int = 30
    parsing_failure: int = 3
    repeating_action: int = 3


class ProgramHTML(TypedDict):
    url: str
    locator: str
    required_contents: str


class ReferenceAnswers(TypedDict):
    exact_match: str
    must_include: list[str]
    fuzzy_match: list[str]


@dataclass
class EvaluatorConfig:
    eval_types: list[str]
    reference_answers: ReferenceAnswers
    reference_url: str
    program_html: list[ProgramHTML]
    url_note: str = "EXACT"
    string_note: Optional[str] = None
    reference_answer_raw_annotation: Optional[str] = None


@dataclass
class ExperimentConfig(ABC):
    goal: str
    task_id: int
    eval: EvaluatorConfig


@dataclass
class BashExperimentConfig(ExperimentConfig):
    goal: str
    task_id: int
    eval: EvaluatorConfig

    require_reset: bool
    setup_commands: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BashExperimentConfig":
        return cls(
            goal=data["intent"],
            task_id=data["task_id"],
            require_reset=data["require_reset"],
            eval=EvaluatorConfig(**data["eval"]),
            setup_commands=data.get("setup_commands", None),
        )

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "task_id": self.task_id,
            "require_reset": self.require_reset,
            "eval": asdict(self.eval),
            "setup_commands": self.setup_commands,
        }


@dataclass
class BrowserExperimentConfig(ExperimentConfig):
    goal: str
    task_id: int
    eval: EvaluatorConfig

    sites: list[str]
    require_login: bool
    storage_state: str
    start_url: str
    geolocation: Optional[str]
    intent_template: Optional[str]
    instantiation_dict: Optional[dict[str, str]]
    require_reset: bool
    intent_template_id: Optional[int]

    # hardcoding some settings that were in args
    headless: bool = True
    slow_mo: int = 0
    observation_type: str = "all"
    observation_type: str = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    save_trace_enabled: bool = True
    sleep_after_execution: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserExperimentConfig":
        """TODO: add the hardcoded args to from_dict if we want to change them"""
        return cls(
            goal=data["intent"],
            task_id=data["task_id"],
            sites=data["sites"],
            require_login=data["require_login"],
            storage_state=data["storage_state"],
            start_url=data["start_url"],
            geolocation=data.get("geolocation", None),
            intent_template=data.get("intent_template", None),
            instantiation_dict=data.get("instantiation_dict", None),
            require_reset=data["require_reset"],
            eval=EvaluatorConfig(**data["eval"]),
            intent_template_id=data.get("intent_template_id", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """TODO: add the hardcoded args to to_dict if we want to record them"""
        return {
            "intent": self.goal,
            "sites": self.sites,
            "task_id": self.task_id,
            "require_login": self.require_login,
            "storage_state": self.storage_state,
            "start_url": self.start_url,
            "geolocation": self.geolocation,
            "intent_template": self.intent_template,
            "instantiation_dict": self.instantiation_dict,
            "require_reset": self.require_reset,
            "eval": asdict(self.eval),
            "intent_template_id": self.intent_template_id,
        }


@dataclass
class BashBrowserExperimentConfig(ExperimentConfig):
    # base args
    goal: str
    task_id: int
    eval: EvaluatorConfig
    # browser args
    sites: list[str]
    require_login: bool
    storage_state: str
    start_url: str
    geolocation: Optional[str]
    intent_template: Optional[str]
    instantiation_dict: Optional[dict[str, str]]
    intent_template_id: Optional[int]
    # bash args
    require_reset: bool
    setup_commands: Optional[list[str]] = None

    # hardcoding some settings that were in args
    headless: bool = True
    slow_mo: int = 0
    observation_type: str = "all"
    observation_type: str = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    save_trace_enabled: bool = True
    sleep_after_execution: float = 0.5

    def to_separate_configs(self) -> tuple[BashExperimentConfig, BrowserExperimentConfig]:
        """Return a BashConfig and BrowserConfig with the data from this config"""
        bash_config = BashExperimentConfig(
            goal=self.goal,
            task_id=self.task_id,
            require_reset=self.require_reset,
            eval=self.eval,
            setup_commands=self.setup_commands,
        )
        browser_config = BrowserExperimentConfig(
            goal=self.goal,
            task_id=self.task_id,
            sites=self.sites,
            require_login=self.require_login,
            storage_state=self.storage_state,
            start_url=self.start_url,
            geolocation=self.geolocation,
            intent_template=self.intent_template,
            instantiation_dict=self.instantiation_dict,
            require_reset=self.require_reset,
            eval=self.eval,
            intent_template_id=self.intent_template_id,
        )
        return bash_config, browser_config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BashBrowserExperimentConfig":
        return cls(
            goal=data["intent"],
            task_id=data["task_id"],
            eval=EvaluatorConfig(**data["eval"]),
            require_reset=data["require_reset"],
            setup_commands=data.get("setup_commands", None),
            sites=data["sites"],
            require_login=data["require_login"],
            storage_state=data["storage_state"],
            start_url=data["start_url"],
            geolocation=data.get("geolocation", None),
            intent_template=data.get("intent_template", None),
            instantiation_dict=data.get("instantiation_dict", None),
            intent_template_id=data.get("intent_template_id", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.goal,
            "task_id": self.task_id,
            "eval": asdict(self.eval),
            "require_reset": self.require_reset,
            "setup_commands": self.setup_commands,
            "sites": self.sites,
            "require_login": self.require_login,
            "storage_state": self.storage_state,
            "start_url": self.start_url,
            "geolocation": self.geolocation,
            "intent_template": self.intent_template,
            "instantiation_dict": self.instantiation_dict,
            "intent_template_id": self.intent_template_id,
        }
