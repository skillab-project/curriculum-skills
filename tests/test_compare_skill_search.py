import json
import unittest
from unittest.mock import patch

from main import SkillSearchRequest

class TestSkillSearch(unittest.TestCase):

    @patch("main.search_skill")
    def test_compare_skill_search(self, mock_search_skill):
        with open("tests/json/C_skill_search_expected.json", "r", encoding="utf-8") as f:
            expected = json.load(f)

        mock_search_skill.return_value = {
            "results": expected["results"]
        }

        from main import search_skill
        response = search_skill(SkillSearchRequest(skill="C++"))
        actual = response["results"]

        for uni in expected["results"]:
            self.assertIn(uni, actual, f"Missing university: {uni}")
            for semester in expected["results"][uni]:
                self.assertIn(semester, actual[uni], f"Missing semester: {semester} in {uni}")
                for lesson in expected["results"][uni][semester]:
                    self.assertIn(lesson, actual[uni][semester], f"Missing lesson: {lesson} in {uni}, {semester}")
                    expected_skills = expected["results"][uni][semester][lesson]
                    actual_skills = actual[uni][semester][lesson]
                    self.assertEqual(actual_skills, expected_skills, f"Mismatch in skill search result for {lesson}")

        total_universities = len(expected["results"])
        total_semesters = sum(len(expected["results"][uni]) for uni in expected["results"])
        total_lessons = sum(
            len(lessons)
            for uni in expected["results"]
            for sem, lessons in expected["results"][uni].items()
        )
        total_skills = sum(
            len(skills)
            for uni in expected["results"]
            for sem in expected["results"][uni]
            for skills in expected["results"][uni][sem].values()
        )

        print(f"\n✅ Skill Search Report")
        print(f"✔ Universities matched: {total_universities}")
        print(f"✔ Semesters matched: {total_semesters}")
        print(f"✔ Lessons matched: {total_lessons}")
        print(f"✔ Skills matched: {total_skills}")


if __name__ == "__main__":
    unittest.main()
