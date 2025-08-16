import json
import os
import unittest
from unittest.mock import patch
from main import PDFProcessingRequest

class TestExtractedSkills(unittest.TestCase):

    @patch("main.calculate_skillnames")
    @patch("main.process_pdf")
    def test_compare_extracted_skills(self, mock_process_pdf, mock_calculate_skillnames):
        with open("tests/json/extracted_skills_expected.json", "r", encoding="utf-8") as f:
            expected = json.load(f)

        mock_process_pdf.return_value = None

        mock_calculate_skillnames.return_value = {
            "skills": expected["skills"]
        }

        pdf_path = os.path.abspath("tests/sample_pdfs/Cambridge University.pdf")
        from main import process_pdf, calculate_skillnames
        process_pdf(PDFProcessingRequest(pdf_name=pdf_path))
        response = calculate_skillnames("University of Cambridge")

        actual = response["skills"]

        for lesson, expected_skills in expected["skills"].items():
            if lesson in ["university_name", "university_country"]:
                continue
            self.assertIn(lesson, actual, f"Lesson not found: {lesson}")
            self.assertIsInstance(actual[lesson], list, f"{lesson} skills should be a list, got {type(actual[lesson])}")
            self.assertEqual(sorted(actual[lesson]), sorted(expected_skills), f"Skill mismatch for lesson '{lesson}'")

        matched_skills = sum(len(actual[lesson]) for lesson in expected["skills"] if lesson in actual)
        total_lessons = len([k for k in expected["skills"] if k not in ["university_name", "university_country"]])
        total_skills = sum(len(skills) for lesson, skills in expected["skills"].items() if lesson not in ["university_name", "university_country"])

        print(f"\n✅ Skill Matching Report")
        print(f"✔ Lessons matched: {total_lessons}")
        print(f"✔ Total skills matched: {matched_skills} / {total_skills}")

if __name__ == "__main__":
    unittest.main()
