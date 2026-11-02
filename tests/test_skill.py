import tempfile
import unittest
from pathlib import Path

from src.core.skill import Skill


class SkillFrontmatterTests(unittest.TestCase):
    def _skill(self, frontmatter: str) -> Skill:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill.md"
            path.write_text(f"---\n{frontmatter}\n---\n\n1. Do it\n")
            return Skill(path)

    def test_yaml_values(self):
        skill = self._skill(
            'name: brief\ndescription: "Quoted: text"\ntools:\n  - read_file\n  - write_file'
        )
        self.assertEqual(skill.description, "Quoted: text")
        self.assertEqual(skill.required_tools, ["read_file", "write_file"])
        self.assertEqual(skill.instructions, "1. Do it")

    def test_inline_tool_list(self):
        skill = self._skill("name: brief\ndescription: Short\ntools: [read_file, write_file]")
        self.assertEqual(skill.required_tools, ["read_file", "write_file"])

    def test_invalid_yaml_falls_back_to_lines(self):
        skill = self._skill("name: brief\ndescription: Use: this tool\ntools: read_file, write_file")
        self.assertEqual(skill.description, "Use: this tool")
        self.assertEqual(skill.required_tools, ["read_file", "write_file"])

    def test_missing_description_is_rejected(self):
        with self.assertRaises(ValueError):
            self._skill("name: brief")


if __name__ == "__main__":
    unittest.main()
