import json
import os
from pathlib import Path

from evals.elsuite.skill_acquisition.eval import SkillAcquisition
from evals.elsuite.skill_acquisition.utils import (
    answer_detected,
    process_answer,
    process_view_instruction,
    view_instruction_detected,
)
from evals.registry import Registry




class TestSkillAcquisition():
    def setup_class(self):
        os.environ["OPENAI_API_KEY"] = "test"
        self.registry = Registry()
        self.dummy_eval_spec = {
            "eval_registry_path": Path("evals/registry"),
            "completion_fns": [self.registry.make_completion_fn("gpt-4")],
            "samples_jsonl": "skill_acquisition/miskito/miskito_test.jsonl",
            "target_language": "miskito",
            "n_samples": 5,
            "knowledge_base_directory": "skill_acquisition/miskito/knowledge_base/",
            "max_replies": 50,
        }

    def test_answer_detected(self):
        assert answer_detected("[ANSWER foo]") is True
        assert answer_detected("[ANSWER: foo]") is True
        assert answer_detected("ANSWER foo") is False
        assert answer_detected("[ANSWER foo") is False
        assert answer_detected("ANSWER foo]") is False
        assert answer_detected("[ANSWER foo][ANSWER bar]") is True


    def test_view_instruction_detected(self):
        SkillAcquisition(**self.dummy_eval_spec)
        assert view_instruction_detected("[VIEW file1]") is True
        assert view_instruction_detected("[VIEW: file1]") is True
        assert view_instruction_detected("[VIEW file1 section1]") is True
        assert view_instruction_detected("[VIEW: file1 section1]") is True
        assert view_instruction_detected("VIEW file1") is False
        assert view_instruction_detected("[VIEW file1") is False
        assert view_instruction_detected("VIEW file1]") is False
        assert view_instruction_detected("[VIEW file1][VIEW file2]") is True
        assert view_instruction_detected("[VIEW: file1][VIEW: file2]") is True


    def test_process_answer(self):
        SkillAcquisition(**self.dummy_eval_spec)
        assert process_answer("[ANSWER foo]") == "foo"
        assert process_answer("[ANSWER: foo]") == "foo"
        assert process_answer("[ANSWER foo bar baz]") == "foo bar baz"
        assert process_answer("[ANSWER: foo bar baz]") == "foo bar baz"
        assert process_answer("[ANSWER foo][ANSWER bar]") == "bar"
        assert process_answer("[ANSWER foo][ANSWER bar") == "foo"


    def test_process_view_instruction(self):
        SkillAcquisition(**self.dummy_eval_spec)
        assert process_view_instruction("[VIEW file1]") == ("file1", None)
        assert process_view_instruction("[VIEW: file1]") == ("file1", None)
        assert process_view_instruction("[VIEW file1 section1]") == (
            "file1",
            "section1",
        )
        assert process_view_instruction("[VIEW: file1 section1]") == (
            "file1",
            "section1",
        )
        assert process_view_instruction("[VIEW file1][VIEW file2]") == (
            "file2",
            None,
        )
        assert process_view_instruction("[VIEW: file1][VIEW: file2]") == (
            "file2",
            None,
        )
        assert process_view_instruction("[VIEW file1 section1][VIEW file2 section2]") == (
            "file2",
            "section2",
        )


    def test_process_view_instruction_spaces_and_quotes(self):
        assert process_view_instruction("[VIEW file1 sectionpart1 sectionpart2]") == (
            "file1",
            "sectionpart1 sectionpart2",
        )
        assert process_view_instruction("[VIEW file1 sectionpart1 'sectionpart2']") == (
            "file1",
            "sectionpart1 'sectionpart2'",
        )


    def test_view_content(self):
        skill_acquisition_eval = SkillAcquisition(**self.dummy_eval_spec)

        # Create a file to view first.
        filepath = skill_acquisition_eval.knowledge_base_directory / "test_file.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"title": "foo", "content": "Test file contents."}) + "\n")

        content, sections_visible_to_model, sections_viewed = skill_acquisition_eval._view_content(
            "test_file.jsonl"
        )
        assert content == "Table of contents for test_file.jsonl: {'foo'}."
        assert sections_visible_to_model == {"test_file.jsonl": {"foo"}}
        assert sections_viewed == {"test_file.jsonl": {"Table of Contents"}}

        content, sections_visible_to_model, sections_viewed = skill_acquisition_eval._view_content(
            "test_file.jsonl", "foo"
        )
        assert content == "Test file contents."
        assert sections_visible_to_model == {"test_file.jsonl": {"foo"}}
        assert sections_viewed == {"test_file.jsonl": {"Table of Contents", "foo"}}

        os.remove(filepath)
