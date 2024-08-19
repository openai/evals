"""base class for evaluation"""

# answer string match
import logging
import time
from abc import ABC
from typing import Union

import evaluate  # type: ignore[import]
from beartype import beartype
from beartype.door import is_bearable

from evals.elsuite.multistep_web_tasks.webarena.bash_browser_env.bash_browser_env import (
    BashBrowserEnv,
)
from evals.elsuite.multistep_web_tasks.webarena.bash_env.actions import BashStopAction
from evals.elsuite.multistep_web_tasks.webarena.browser_env.actions import BrowserAction
from evals.elsuite.multistep_web_tasks.webarena.browser_env.basic_browser_env import BrowserEnv
from evals.elsuite.multistep_web_tasks.webarena.browser_env.browser_utils import BrowserEnvOutput
from evals.elsuite.multistep_web_tasks.webarena.core.env import (
    Action,
    EnvOutput,
    LLMAgentEnv,
    Trajectory,
)
from evals.elsuite.multistep_web_tasks.webarena.core.utils import (
    BashBrowserExperimentConfig,
    BashExperimentConfig,
    BrowserExperimentConfig,
    ExperimentConfig,
)
from evals.elsuite.multistep_web_tasks.webarena.evaluation_harness.helper_functions import (
    llm_fuzzy_match,
)

logger = logging.getLogger(__name__)


@beartype
class Evaluator(object):
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        trajectory: Trajectory,
        env: LLMAgentEnv,
        experiment_config: ExperimentConfig,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            # i'm confused but seems to work
            is_bearable(trajectory[-1].action, Action)
            last_action = trajectory[-1].action
        except Exception:
            raise ValueError(
                "The last element of trajectory should have an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> EnvOutput:
        try:
            is_bearable(trajectory[-1].env_output, EnvOutput)
            last_state = trajectory[-1].env_output
        except Exception:
            raise ValueError(
                "The last element of trajectory should contain a state, add a fake stop action if needed"
            )

        return last_state  # type: ignore[return-value]


def get_answer_from_action(action: Action) -> str:
    """This is kind of a hack to make BrowserAction and BashAction
    play nicely together"""
    if isinstance(action, BrowserAction):
        return action.data["answer"]
    elif isinstance(action, BashStopAction):
        return action.answer
    else:
        # for early stopping, sometimes we'll want an answer from a non-Stop action
        return ""


@beartype
class StringEvaluator(Evaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    def __call__(
        self,
        trajectory: Trajectory,
        env: LLMAgentEnv,
        experiment_config: ExperimentConfig,
    ) -> float:
        def clean_answer(answer: str) -> str:
            if answer.startswith("'") and answer.endswith("'"):
                answer = answer[1:-1]
            elif answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            return answer.lower()

        last_action = self.get_last_action(trajectory)
        pred = clean_answer(get_answer_from_action(last_action))

        score = 1.0
        reference_answers = experiment_config.eval.reference_answers
        assert reference_answers is not None, "uh oh, why is this None"
        for approach, value in reference_answers.items():
            if approach == "exact_match":
                assert isinstance(value, str)
                ref_answer = clean_answer(value)
                score = score * (pred == ref_answer)
            elif approach == "must_include":
                assert isinstance(value, list)
                for must_value in value:
                    must_value = clean_answer(must_value)
                    score = score * (must_value in pred)
            elif approach == "fuzzy_match":
                intent = experiment_config.goal
                assert isinstance(value, list)
                for reference in value:
                    fuzzy_score = llm_fuzzy_match(pred, reference, intent)
                    score = score * fuzzy_score
        return score


@beartype
class StringSoftEvaluator(Evaluator):
    """Use text generation metrics such as BLEU, ROUGE, etc. to evaluate the answer"""

    def __call__(
        self,
        trajectory: Trajectory,
        env: LLMAgentEnv,
        experiment_config: ExperimentConfig,
    ) -> float:
        last_action = self.get_last_action(trajectory)
        pred = get_answer_from_action(last_action)
        ref = experiment_config.eval.reference_answers
        # rouge
        m = evaluate.load("rouge")
        rouge = m.compute(predictions=[pred], references=[ref])
        return float(rouge["rouge1"])  # type: ignore [ian: i think they work]


class BrowserEvaluator(Evaluator):
    """Subclass specifically for evaluators that only work in the BrowserEnv"""


@beartype
class URLExactEvaluator(BrowserEvaluator):
    """Check whether the URL is exactly the same as of the reference URLs"""

    def __call__(
        self,
        trajectory: Trajectory,
        env: BrowserEnv,
        experiment_config: BrowserExperimentConfig,
    ) -> float:
        def clean_url(url: str) -> str:
            # NOTE: also dropping http:// and https://
            url = str(url)
            if url.endswith("/"):
                url = url[:-1]
            if url.startswith("http://"):
                url = url[7:]
            if url.startswith("https://"):
                url = url[8:]
            return url

        last_state = self.get_last_state(trajectory)
        assert isinstance(last_state, BrowserEnvOutput)
        last_page = last_state.info.page
        pred = clean_url(last_page.url)
        ref_urls = experiment_config.eval.reference_url.split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = experiment_config.eval.url_note
        if matching_rule == "EXACT":
            if pred in ref_urls:
                return 1.0
            else:
                return 0.0
        elif matching_rule == "GOLD in PRED":
            if any([ref in pred for ref in ref_urls]):
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")


@beartype
class HTMLContentExactEvaluator(BrowserEvaluator):
    """Check whether the contents appear in the page"""

    def __call__(
        self,
        trajectory: Trajectory,
        env: Union[BrowserEnv, BashBrowserEnv],
        experiment_config: Union[BrowserExperimentConfig, BashBrowserExperimentConfig],
    ) -> float:
        def clean(text: str) -> str:
            text = str(text)
            return text.strip().lower()

        targets = experiment_config.eval.program_html
        page = env.page

        score = 1.0
        for target in targets:
            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)

            required_contents: str = target["required_contents"]  # what contents to check
            locator: str = target["locator"]  # js element locator

            # navigate to that url
            if target_url != "last":
                page.goto(target_url)
                time.sleep(3)  # TODO [shuyanzh]: fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                selected_element = page.content()
            # use JS to select the element
            elif locator.startswith("document."):
                try:
                    selected_element = page.evaluate(f"() => {locator}")
                    if not selected_element:
                        selected_element = ""
                    selected_element = str(selected_element)
                except Exception as e:
                    # the page is wrong, return empty
                    logger.error(f"Error in evaluating locator {locator}: {e}")
                    selected_element = ""
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = eval(func)
            else:
                raise ValueError(f"Unknown locator: {locator}")

            required_contents_or = [clean(x) for x in required_contents.split(" |OR| ")]
            selected_element = clean(selected_element)
            score *= any([content in selected_element for content in required_contents_or])

        return score


class EvaluatorComb(ABC):
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    def __call__(
        self,
        trajectory: Trajectory,
        env: LLMAgentEnv,
        experiment_config: ExperimentConfig,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(trajectory, env, experiment_config)
            # TODO: work out why it's done this way
            score *= cur_score
        return score


class BrowserEvaluatorComb(EvaluatorComb):
    def __init__(self, evaluators: list[BrowserEvaluator]) -> None:
        self.evaluators = evaluators


@beartype
def evaluator_router(experiment_config: ExperimentConfig) -> EvaluatorComb:
    if isinstance(experiment_config, BrowserExperimentConfig):
        assert isinstance(experiment_config, BrowserExperimentConfig)
        return browser_evaluator_router(experiment_config)
    elif isinstance(experiment_config, BashExperimentConfig):
        assert isinstance(experiment_config, BashExperimentConfig)
        return bash_evaluator_router(experiment_config)
    elif isinstance(experiment_config, BashBrowserExperimentConfig):
        assert isinstance(experiment_config, BashBrowserExperimentConfig)
        return bash_browser_evaluator_router(experiment_config)
    else:
        raise ValueError(f"Unknown experiment_config type {type(experiment_config)}")


@beartype
def browser_evaluator_router(
    experiment_config: BrowserExperimentConfig,
) -> EvaluatorComb:
    """Router to get the evaluator class"""

    # TODO: add 'eval' and maybe others to the experiment_config base class
    eval_types = experiment_config.eval.eval_types
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        if eval_type == "string_match":
            evaluators.append(StringEvaluator())
        elif eval_type == "url_match":
            evaluators.append(URLExactEvaluator())
        elif eval_type == "program_html":
            evaluators.append(HTMLContentExactEvaluator())
        else:
            raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)


def bash_evaluator_router(
    experiment_config: BashExperimentConfig,
) -> EvaluatorComb:
    """Router to get the evaluator class"""
    # TODO: add 'eval' and maybe others to the experiment_config base class
    eval_types = experiment_config.eval.eval_types
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        if eval_type == "string_match":
            evaluators.append(StringEvaluator())
        else:
            raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)


def bash_browser_evaluator_router(
    experiment_config: BashBrowserExperimentConfig,
) -> EvaluatorComb:
    """Router to get the evaluator class"""
    # TODO: add 'eval' and maybe others to the experiment_config base class
    eval_types = experiment_config.eval.eval_types
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        if eval_type == "string_match":
            evaluators.append(StringEvaluator())
        elif eval_type == "url_match":
            evaluators.append(URLExactEvaluator())
        elif eval_type == "program_html":
            evaluators.append(HTMLContentExactEvaluator())
        else:
            raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)
