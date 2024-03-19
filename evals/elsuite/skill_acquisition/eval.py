import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.elsuite.skill_acquisition.task_description import TASK_DESCRIPTION
from evals.elsuite.skill_acquisition.utils import (
    PROMPTS,
    answer_detected,
    get_accuracy,
    get_average_bleu_score,
    get_average_invalid_retrieval_calls,
    get_average_retrieval_calls,
    get_average_retrieval_precision,
    get_bleu_score,
    get_bootstrap_accuracy_std,
    get_question_type,
    get_std_of_difference,
    process_answer,
    process_view_instruction,
    render_intermediate_prompt,
    view_instruction_detected,
)
from evals.eval import SolverEval
from evals.solvers.solver import Solver
from evals.task_state import Message, TaskState

TARGET_LANGUAGES = ["miskito"]
LESSON_FILE_SUFFIX = "_lessons.jsonl"

logger = logging.getLogger(__name__)


class SkillAcquisition(SolverEval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        samples_jsonl: str,
        target_language: str,
        knowledge_base_directory: str,
        max_replies: int,
        seed: int = 6122023,
        n_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, seed=seed, *args, **kwargs)

        assert (
            target_language.lower() in TARGET_LANGUAGES
        ), f"Error: target language must be one of {TARGET_LANGUAGES}"

        self.samples_jsonl = samples_jsonl
        self.n_samples = n_samples
        self.task_description = TASK_DESCRIPTION.format(target_language=target_language)
        self.rng = random.Random(seed)

        # Retrieval-related attributes.
        self.knowledge_base_directory = self._prefix_registry_path(knowledge_base_directory)
        self.files_available = os.listdir(self.knowledge_base_directory)
        self.content_by_file: dict[str, dict] = {}
        self.max_replies = max_replies  # Used as timeout.

    def eval_sample(self, solver: Solver, sample: Dict, rng: random.Random) -> Dict[str, Any]:
        """Runs the appropriate private evaluation function depending on the eval phase: retrieval or non-retrieval.

        Args:
            solver (Solver): per-sample solver instantiated in parent.
            sample (Dict): input to evaluate on.
            rng (random.Random): random number generator, used for reproducibility.

        Returns:
            Dict[str, Any]: metrics collected during evaluation.
        """
        # since we run two discrete experiments per sample, we have to copy the solver ahead of time
        non_retrieval_solver = solver.copy()
        retrieval_solver = solver.copy()
        non_retrieval_out = self._eval_non_retrieval_sample(non_retrieval_solver, sample)
        retrieval_out = self._eval_retrieval_sample(retrieval_solver, sample)
        metrics_obj = {
            "non_retrieval": non_retrieval_out,
            "retrieval": retrieval_out,
        }

        evals.record.record_metrics(**metrics_obj)
        return metrics_obj

    def _eval_non_retrieval_sample(self, solver: Solver, sample: Dict, *_) -> Dict[str, Any]:
        """Evaluates the given sample without using retrieval, ie. using the solver directly.

        Args:
            solver (Solver): any compatible solver, instantiated just for this sample.
            sample (Dict): input to evaluate on.

        Returns:
            Dict[str, Any]: metrics collected during evaluation.
        """
        task_state = TaskState(
            task_description=self.task_description,
            messages=[Message(**msg) for msg in sample["input"]],
        )

        result = solver(task_state)
        output = result.output
        if answer_detected(output):
            answer = process_answer(output)
            logger.debug(f"Model answered {answer}")
        else:
            answer = "NO ANSWER DETECTED"

        picked = evals.record_and_check_match(
            prompt=sample["input"],
            sampled=answer,
            expected=[sample["ideal"]],
        )

        out_obj = {
            "prompt": sample["input"],
            "raw_output": result.output,
            "parsed_output": answer,
            "expected": [sample["ideal"]],
            "correct": picked is not None,
            "bleu": get_bleu_score(sample["ideal"], answer),
            "question_type": get_question_type(sample["input"][-1]["content"]),
        }
        return out_obj

    def _eval_retrieval_sample(self, solver: Solver, sample: Dict, *_) -> Dict[str, Any]:
        """Evaluates the given sample using retrieval. The retrieval logic is implemented in the _conversation_loop function.

        Args:
            solver (Solver): any compatible solver, instantiated just for this sample.
            sample (Dict): input to evaluate on.

        Returns:
            Dict[str, Any]: metrics collected during evaluation.
        """
        files_available_paths = [
            self.knowledge_base_directory / file for file in self.files_available
        ]
        assert all([file.exists() for file in files_available_paths])
        task_state = TaskState(
            task_description=self.task_description,
            messages=[Message(**msg) for msg in sample["input"]],
            current_state={"files": files_available_paths},
        )

        output, metrics = self._conversation_loop(solver, task_state)

        if answer_detected(output):
            answer = process_answer(output)
            logging.debug(f"Model answered {answer}")
        elif output == "Context length exceeded.":
            answer = "NO ANSWER DETECTED"
            logger.warn("Current interaction exceeded model context length.")
        else:
            answer = "NO ANSWER DETECTED"
            logging.debug(f"Model timed out after {metrics['current_replies']} replies.")

        picked = evals.record_and_check_match(
            prompt=sample["input"],
            sampled=answer,
            expected=[sample["ideal"]],
        )

        out_obj = {
            "prompt": sample["input"],
            "raw_output": output,
            "parsed_output": answer,
            "expected": [sample["ideal"]],
            "correct": picked is not None,
            "bleu": get_bleu_score(sample["ideal"], answer),
            "ctx_len_exceeded": output == "Context length exceeded.",
            "interaction_timed_out": metrics["current_replies"] >= self.max_replies,
            "question_type": get_question_type(sample["input"][-1]["content"]),
            "lesson_retrieval_calls": metrics["lesson_retrieval_calls"],
            "correct_retrieval_calls": metrics["correct_retrieval_calls"],
            "invalid_retrieval_calls": metrics["total_retrieval_calls"]
            - metrics["correct_retrieval_calls"],
            "total_retrieval_calls": metrics["total_retrieval_calls"],
        }
        return out_obj

    def run(self, recorder: evals.record.Recorder) -> dict[str, Union[float, int]]:
        samples = self.get_samples()
        self.rng.shuffle(samples)
        samples = samples[: self.n_samples] if self.n_samples is not None else samples

        results = self.eval_all_samples(recorder, samples)
        non_retrieval_results = [result["non_retrieval"] for result in results]
        retrieval_results = [result["retrieval"] for result in results]

        baseline_accuracy = get_accuracy(non_retrieval_results)
        baseline_std = get_bootstrap_accuracy_std(non_retrieval_results)

        retrieval_accuracy = get_accuracy(retrieval_results)
        retrieval_std = get_bootstrap_accuracy_std(retrieval_results)

        delta_accuracy = retrieval_accuracy - baseline_accuracy

        # TODO: decide which metric to report – propagated standard deviation
        # from bootstrapping or standard error of the mean estimated from repeats
        # of the eval experiments.
        delta_std = get_std_of_difference(baseline_std, retrieval_std)

        ctx_len_exceeded_rate = sum(
            1 for result in retrieval_results if result["ctx_len_exceeded"]
        ) / len(retrieval_results)
        timeout_rate = sum(
            1 for result in retrieval_results if result["interaction_timed_out"]
        ) / len(retrieval_results)

        num_translation_samples = len(
            [result for result in retrieval_results if result["question_type"] == "translation"]
        )
        num_non_translation_samples = len(
            [result for result in retrieval_results if result["question_type"] == "non-translation"]
        )

        result = {
            "baseline_accuracy": baseline_accuracy,
            "baseline_std": baseline_std,
            "retrieval_accuracy": retrieval_accuracy,
            "retrieval_std": retrieval_std,
            "delta_accuracy": delta_accuracy,
            "delta_std": delta_std,
            "average_retrieval_precision": get_average_retrieval_precision(retrieval_results),
            "average_non_retrieval_bleu_score": get_average_bleu_score(non_retrieval_results),
            "average_retrieval_bleu_score": get_average_bleu_score(retrieval_results),
            "average_retrieval_calls": get_average_retrieval_calls(retrieval_results),
            "average_invalid_retrieval_calls": get_average_invalid_retrieval_calls(
                retrieval_results
            ),
            "ctx_len_exceeded_rate": ctx_len_exceeded_rate,
            "timeout_rate": timeout_rate,
            "num_samples": len(retrieval_results),
            "num_translation_samples": num_translation_samples,
            "num_non_translation_samples": num_non_translation_samples,
        }

        return result

    def _view_content(
        self,
        file_name: str,
        section_title: str = None,
        sections_visible_to_model: dict[str, set] = defaultdict(set),
        sections_viewed: dict[str, set] = defaultdict(set),
    ) -> tuple[str, dict[str, set], dict[str, set]]:
        """Views content from a JSONL file in the knowledge base.
        If a section is provided, only the contents of that section are returned.
        If no section is specified, the function returns the table of contents of the file.

        Args:
            file_name (str): Name of the file. Full directory prefixed automatically.
            section_title (str, optional): Name of the section to view. Defaults to None.
            sections_visible_to_model (dict[str, set], optional): Dictionary of sections visible to the model. Defaults to {}. Updated in-place.
            sections_viewed (dict[str, set], optional): Dictionary of sections viewed by the model. Defaults to {}. Updated in-place.

        Returns:
            tuple(str, dict[str, set], dict[str, set]): A tuple of
                the content of the section (if specified) and
                the updated dictionaries of sections visible to and viewed by the model.
        """
        # TODO: more general file format.

        if file_name in self.content_by_file:
            file_content_by_section = self.content_by_file[file_name]
        else:
            # This should never occur, but if it does it should stop the eval from running.
            if not os.path.exists(self.knowledge_base_directory / file_name):
                raise ValueError(
                    f"File {self.knowledge_base_directory / file_name} does not exist."
                )

            file_content_by_section = {}
            with open(self.knowledge_base_directory / file_name, "r") as f:
                for line in f:
                    line_dict = json.loads(line)
                    file_content_by_section[line_dict["title"]] = line_dict["content"]
            self.content_by_file[file_name] = file_content_by_section

        if section_title is None:
            sections = set(file_content_by_section.keys())
            sections_visible_to_model[file_name] = sections
            sections_viewed[file_name].add("Table of Contents")

            return (
                f"Table of contents for {file_name}: {sections}.",
                sections_visible_to_model,
                sections_viewed,
            )

        sections_viewed[file_name].add(section_title)
        return file_content_by_section[section_title], sections_visible_to_model, sections_viewed

    def _conversation_loop(
        self, solver: Solver, task_state: TaskState
    ) -> tuple[str, Dict[str, int]]:
        """Maintains a conversation with the model until it outputs an answer or times out.
        The model may request to read a file or a section of a file from the knowledge base.

        Args:
            solver (Solver): any compatible solver, instantiated just for this sample.
            task_state (TaskState): current task_state, which additionally contains a list of knowledge base files in `current_state`.

        Returns:
            tuple[str, Dict[str, int]]: a tuple of the model's output and a dictionary of metrics collected during the conversation.
        """
        output = ""

        # Not all retrieval calls are valid, e.g. if the file doesn't exist.
        # These two metrics are analogous to an instruction-following rate.
        metrics = {
            "lesson_retrieval_calls": 0,
            "correct_retrieval_calls": 0,
            "total_retrieval_calls": 0,
            "current_replies": 0,
        }
        sections_visible_to_model: dict[str, set] = defaultdict(set)
        sections_viewed: dict[str, set] = defaultdict(set)
        consecutive_instruction_failures = 0

        while not answer_detected(output) and metrics["current_replies"] < self.max_replies:
            if metrics["current_replies"] == 0:
                # Beginning of the conversation, prepare instructions.
                task_state.task_description = (
                    task_state.task_description
                    + "\n\n"
                    + PROMPTS["retrieval_instructions"].format(list_of_files=self.files_available)
                )
            if len(sections_viewed.items()) > 0:
                intermediate_prompt = render_intermediate_prompt(sections_viewed)
                task_state.messages += [Message(role="system", content=intermediate_prompt)]

            output = solver(task_state).output
            task_state.messages += [Message(role="assistant", content=output)]
            metrics["current_replies"] += 1

            if view_instruction_detected(output) or answer_detected(output):
                consecutive_instruction_failures = 0

            if view_instruction_detected(output):
                file, section = process_view_instruction(output)
                metrics["total_retrieval_calls"] += 1

                if file.endswith(LESSON_FILE_SUFFIX):
                    metrics["lesson_retrieval_calls"] += 1

                # Handle any errors by logging and re-prompting the model.
                if file not in self.files_available:
                    task_state.messages += [
                        Message(
                            role="system",
                            content=PROMPTS["wrong_file"].format(
                                file=file, knowledge_base=self.files_available
                            ),
                        )
                    ]
                    logger.debug(
                        f"Model tried to view {file}, which does not exist in the knowledge base:\n{json.dumps(self.files_available, indent=4)}."
                    )
                    continue

                if section is not None and section not in sections_visible_to_model[file]:
                    task_state.messages += [
                        Message(
                            role="system",
                            content=PROMPTS["wrong_section"].format(
                                file=file,
                                section=section,
                                table_of_contents=sections_visible_to_model[file],
                            ),
                        )
                    ]
                    logger.debug(
                        f"Model tried to view section {section} in file {file}, which does not exist.\nAvailable sections are {json.dumps(list(sections_visible_to_model[file]), indent=4)}."
                    )
                    continue

                # If no errors, view the content and update the task state.
                content, sections_visible_to_model, sections_viewed = self._view_content(
                    file, section, sections_visible_to_model, sections_viewed
                )
                task_state.messages += [
                    Message(
                        role="system",
                        content=PROMPTS["present_content"].format(
                            file=file,
                            section=section if section is not None else "Table of Contents",
                            content=content,
                        ),
                    ),
                ]
                metrics["correct_retrieval_calls"] += 1
                if section is None:
                    logger.debug(f"Model viewed table of contents for file {file}: {content}")
                else:
                    logger.debug(f"Model viewed section {section} in file {file}.")
            elif not answer_detected(output):
                if consecutive_instruction_failures >= 3:
                    return "Model failed to follow instructions.", metrics

                consecutive_instruction_failures += 1
                logger.debug(
                    f"Model output did not contain a view instruction or an answer: {output}"
                )

                # Flag & move onto next sample if context length exceeded.
                if (
                    "'code': 'context_length_exceeded'" in output
                    or "Please reduce your prompt; or completion length" in output
                ):
                    return "Context length exceeded.", metrics

                task_state.messages += [
                    Message(
                        role="system",
                        content="Your output did not contain a view instruction or an answer. Please try again.",
                    )
                ]

        return output, metrics
