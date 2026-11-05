"""Score parsing and saved-run integrity; real GPU batching is smoke-tested separately."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from benchmark_steering import GENERATION_FIELDS, JUDGMENT_FIELDS, load_judging_state
from llm.judge import _parse_judge_output


class JudgeParsingTests(unittest.TestCase):
    def test_reasoning_scores_do_not_override_the_verdict(self):
        raw = '<think>Rejected answer: <score>no</score>\nThe topic is present.</think><score> YES </score>'
        verdict = _parse_judge_output(raw)
        self.assertEqual(verdict["label"], "yes")
        self.assertTrue(verdict["meets_criterion"])
        self.assertEqual(verdict["reasoning"], 'Rejected answer: <score>no</score>\nThe topic is present.')
        self.assertEqual(verdict["raw"], raw)

    def test_negative_verdict_without_reasoning(self):
        verdict = _parse_judge_output('<score> no </score><|end_of_text|>')
        self.assertEqual(verdict["label"], "no")
        self.assertFalse(verdict["meets_criterion"])
        self.assertEqual(verdict["reasoning"], "")

    def test_invalid_or_ambiguous_output_is_not_counted_as_no(self):
        for raw in (
            "yes", "", '<score>maybe</score>',
            '<think><score>yes</score></think>',
            '<score>yes</score><score>no</score>',
        ):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                _parse_judge_output(raw)


class JudgingResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.coherence = "Saved readability criterion, not the current default."
        self.concepts = {"dog": "Saved dog criterion.", "radio": "Saved radio criterion."}
        self.rows = [
            self.generation("dog", "unsteered", "", "0", ', a dog barked,\nthen wagged "its" tail.'),
            self.generation("dog", "unsteered", "", "1", ', a dog barked,\nthen wagged "its" tail.'),
            self.generation("radio", "identity", "1.0", "0", ', the radio station broadcast the weather.'),
        ]
        self.write_csv(self.root / "generations.csv", GENERATION_FIELDS, self.rows)

    @staticmethod
    def generation(concept, method, strength, trial, completion):
        prompt = "Once upon a time"
        return dict(zip(GENERATION_FIELDS, (
            concept, method, strength, trial, "0", prompt, completion, prompt + completion,
        )))

    @staticmethod
    def judgment(row, coherence="yes", concept="yes"):
        return {
            **row,
            "coherence_label": coherence, "concept_label": concept,
            "coherence_reasoning": "", "concept_reasoning": "",
            "coherence_raw": f'<think>\n</think>\n<score>{coherence}</score>',
            "concept_raw": f'<think>\n</think>\n<score>{concept}</score>',
        }

    @staticmethod
    def write_csv(path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def save_judgments(self, rows):
        path = self.root / "judgments.csv"
        self.write_csv(path, GENERATION_FIELDS + JUDGMENT_FIELDS, rows)
        return path

    def load(self):
        return load_judging_state(
            self.root, coherence_criterion=self.coherence, concept_criteria=self.concepts,
        )

    def test_resume_keeps_completed_records_and_duplicate_draws(self):
        completed = self.judgment(self.rows[0])
        path = self.save_judgments([completed])
        original = path.read_bytes()
        generations, saved, cache = self.load()
        self.assertEqual(generations, self.rows)
        self.assertEqual(saved, [completed])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(cache[(self.coherence, self.rows[0]["prompt"], self.rows[0]["completion"])]["label"], "yes")
        self.assertEqual(cache[(self.concepts["dog"], self.rows[0]["prompt"], self.rows[0]["completion"])]["label"], "yes")
        self.assertEqual([row["trial"] for row in generations[:2]], ["0", "1"])

    def test_only_an_unfinished_trailing_record_is_removed(self):
        completed = self.judgment(self.rows[0])
        path = self.save_judgments([completed])
        prefix = path.read_bytes()
        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=GENERATION_FIELDS + JUDGMENT_FIELDS).writerow(self.judgment(self.rows[2]))
        tail = buffer.getvalue().encode("utf-8")
        with path.open("ab") as stream:
            stream.write(tail[:tail.index(b"<score>") + 4])
        generations, saved, _ = self.load()
        self.assertEqual(saved, [completed])
        self.assertEqual(generations, self.rows)
        self.assertEqual(path.read_bytes(), prefix)
        self.assertEqual(self.load()[1], [completed])

    def test_complete_final_record_without_newline_is_kept(self):
        completed = self.judgment(self.rows[0])
        path = self.save_judgments([completed])
        original = path.read_bytes().rstrip(b"\r\n")
        path.write_bytes(original)
        self.assertEqual(self.load()[1], [completed])
        self.assertEqual(path.read_bytes(), original)

    def test_interrupted_utf8_tail_keeps_all_completed_records(self):
        completed = self.judgment(self.rows[0])
        path = self.save_judgments([completed])
        prefix = path.read_bytes()
        buffer = io.StringIO(newline="")
        unfinished = {**self.judgment(self.rows[2]), "concept_reasoning": "Broadcast \u96fb"}
        csv.DictWriter(buffer, fieldnames=GENERATION_FIELDS + JUDGMENT_FIELDS).writerow(unfinished)
        tail = buffer.getvalue().encode("utf-8")
        cut = tail.index("\u96fb".encode("utf-8")) + 2
        with path.open("ab") as stream:
            stream.write(tail[:cut])
        self.assertEqual(self.load()[1], [completed])
        self.assertEqual(path.read_bytes(), prefix)

    def test_complete_invalid_labels_are_rejected_without_truncation(self):
        path = self.save_judgments([self.judgment(self.rows[0], coherence="maybe")])
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(path.read_bytes(), original)

    def test_changed_generation_or_duplicate_identity_is_rejected(self):
        completed = self.judgment(self.rows[0])
        changed = {**completed, "completion": "A different response."}
        for records in ([changed], [completed, completed]):
            with self.subTest(records=records):
                path = self.save_judgments(records)
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    self.load()
                self.assertEqual(path.read_bytes(), original)

    def test_conflicting_cached_verdicts_are_rejected(self):
        path = self.save_judgments([
            self.judgment(self.rows[0]), self.judgment(self.rows[1], coherence="no"),
        ])
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(path.read_bytes(), original)

    def test_malformed_middle_record_does_not_discard_later_results(self):
        path = self.save_judgments([self.judgment(self.rows[0])])
        with path.open("a", newline="", encoding="utf-8") as stream:
            stream.write("dog,unsteered\n")
            csv.DictWriter(stream, fieldnames=GENERATION_FIELDS + JUDGMENT_FIELDS).writerow(self.judgment(self.rows[2]))
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(path.read_bytes(), original)

    def test_duplicate_generation_identity_is_rejected(self):
        self.write_csv(self.root / "generations.csv", GENERATION_FIELDS, [*self.rows, self.rows[0]])
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
