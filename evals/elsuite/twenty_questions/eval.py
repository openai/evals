import logging
import random
import re
from typing import Any, Dict, List, Optional, Union

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.twenty_questions.utils import PROMPTS, generate_task_state_for
from evals.eval import SolverEval
from evals.record import Recorder
from evals.registry import registry
from evals.solvers.human_cli_solver import HumanCliSolver
from evals.solvers.solver import Solver
from evals.solvers.utils import maybe_wrap_with_solver
from evals.task_state import Message

logger = logging.getLogger(__name__)
WORD_PATTERN = r"\[GUESS (.*?)\]"


class TwentyQuestions(SolverEval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        samples_jsonl: str,
        gamemaster_spec: str,
        max_questions: int = 20,
        max_replies: int = 40,
        num_shortlist_items: int = 20,
        shortlist_variant: bool = False,
        seed: int = 222024,
        n_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, seed=seed, *args, **kwargs)

        self.samples_jsonl = samples_jsonl
        self.gamemaster_solver = maybe_wrap_with_solver(
            registry.make_completion_fn(gamemaster_spec)
        )
        self.max_questions = max_questions

        if max_replies < max_questions:
            logger.warn(
                f"max_replies ({max_replies}) is less than max_questions ({max_questions}). Setting max_replies to {max_questions + 20}"
            )
        self.max_replies = max_replies if max_replies > max_questions else max_questions + 20
        self.num_shortlist_items = num_shortlist_items
        self.shortlist_variant = shortlist_variant

        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def eval_sample(self, solver: Solver, sample: Dict, rng: random.Random) -> Dict[str, Any]:
        assert "word" in sample, "Sample must contain 'word' field"
        assert "difficulty" in sample, "Sample must contain 'difficulty' field"

        if not isinstance(solver, HumanCliSolver):
            logging.info(f"Running sample: {sample['word']}")

        # Generate the shortlist for the current sample if applicable.
        if self.shortlist_variant:
            assert self.num_shortlist_items <= len(
                self.shortlist
            ), "Number of shortlist items must be less than or equal to the total number of samples."
            shortlist_for_sample = rng.sample(self.shortlist, self.num_shortlist_items)
            if sample["word"] not in shortlist_for_sample:
                random_index = rng.randint(0, len(shortlist_for_sample) - 1)
                shortlist_for_sample[random_index] = sample["word"]
        else:
            shortlist_for_sample = None
        response = self._conversation_loop(solver, sample, shortlist_for_sample)

        return response

    def run(self, recorder: Recorder) -> Dict[str, Union[float, int]]:
        samples = self.get_samples()
        self.rng.shuffle(samples)
        samples = samples[: self.n_samples] if self.n_samples else samples

        if self.shortlist_variant:
            self.shortlist = [sample["word"] for sample in samples]

        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        scores = [event.data["score"] for event in events]
        num_guesses = [event.data["num_guesses"] for event in events]
        num_questions = [event.data["num_questions"] for event in events]
        num_violations = [event.data["num_violations"] for event in events]
        num_gamemaster_refusals = [event.data["num_gamemaster_refusals"] for event in events]
        incorrect_guesses = [event.data["incorrect_guesses"] for event in events]
        word_difficulties = [event.data["word_difficulty"] for event in events]

        return {
            "score": sum(scores) / len(scores),
            "accuracy": evals.metrics.get_accuracy(events),
            "bootstrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
            "average_num_guesses": sum(num_guesses) / len(num_guesses),
            "average_num_questions": sum(num_questions) / len(num_questions),
            "average_num_violations": sum(num_violations) / len(num_violations),
            "average_num_gamemaster_refusals": sum(num_gamemaster_refusals)
            / len(num_gamemaster_refusals),
            "average_num_incorrect_guesses": sum((len(ig) for ig in incorrect_guesses))
            / len(incorrect_guesses),
            "average_word_difficulty": sum(word_difficulties) / len(word_difficulties),
        }

    def _conversation_loop(
        self, solver: Solver, sample: Dict, shortlist: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Maintains a conversation between the guesser and the gamemaster until the maximum number of questions is reached, or until a correct guess is made.

        Args:
            solver (Solver): any compatible solver, instantiated for the current sample.
            sample (Dict): current sample – one word to guess, and its associated difficulty.

        Returns:
            Dict[str, Any]: a dictionary containing the final result and metrics of the conversation.
        """

        metrics = {
            "num_guesses": 0,
            "num_questions": 0,
            "num_violations": 0,
            "num_guesser_replies": 0,  # num_guesses + num_questions + num_violations
            "num_gamemaster_refusals": 0,
            "incorrect_guesses": [],
        }
        conversation = []

        # Contains fall-back condition to avoid infinite loops for solvers which never output questions.
        while (
            metrics["num_questions"] < self.max_questions
            and metrics["num_guesser_replies"] < self.max_replies
        ):
            task_state = generate_task_state_for(
                "guesser", conversation, max_questions=self.max_questions, shortlist=shortlist
            )
            guesser_response = solver(task_state)
            conversation += [Message(content=guesser_response.output, role="guesser")]
            metrics["num_guesser_replies"] += 1

            # Check if guess made:
            match = re.search(WORD_PATTERN, guesser_response.output)
            if match is not None:
                metrics["num_guesses"] += 1
                guess = match.group(1)
                if guess.lower() == sample["word"].lower():
                    response = {
                        "correct": True,
                        "score": self.max_questions - metrics["num_questions"],
                        "expected": sample["word"],
                        "word_difficulty": sample["difficulty"],
                        "picked": guess,
                        "num_guesses": metrics["num_guesses"],
                        "num_questions": metrics["num_questions"],
                        "num_violations": metrics["num_violations"],
                        "num_gamemaster_refusals": metrics["num_gamemaster_refusals"],
                        "incorrect_guesses": metrics["incorrect_guesses"],
                    }
                    evals.record.record_match(**response)
                    return response
                else:
                    metrics["incorrect_guesses"] += [guess]
                    conversation += [
                        Message(
                            content=PROMPTS["incorrect_guess"].format(guess=guess), role="system"
                        )
                    ]
                    continue
            elif "?" in guesser_response.output.strip():
                metrics["num_questions"] += 1
            else:  # Neither guess nor question.
                # TODO: Maybe make the guesser retry here?
                logger.warn(
                    f"Rule violation, no guess or question in output: {guesser_response.output}"
                )
                metrics["num_violations"] += 1
                conversation += [Message(content=PROMPTS["rule_violation"], role="system")]
                continue

            task_state = generate_task_state_for("gamemaster", conversation, sample["word"])
            gamemaster_response = self.gamemaster_solver(task_state)
            conversation += [Message(content=gamemaster_response.output, role="gamemaster")]
            if gamemaster_response.output.lower() == "skip":
                metrics["num_gamemaster_refusals"] += 1

        logger.info(f"Ran out of questions for word: {sample['word']}")
        response = {
            "correct": False,
            "score": 0,
            "expected": sample["word"],
            "word_difficulty": sample["difficulty"],
            "num_guesses": metrics["num_guesses"],
            "num_questions": metrics["num_questions"],
            "num_violations": metrics["num_violations"],
            "num_gamemaster_refusals": metrics["num_gamemaster_refusals"],
            "incorrect_guesses": metrics["incorrect_guesses"],
        }
        evals.record.record_match(**response)
        return response
