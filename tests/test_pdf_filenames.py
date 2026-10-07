"""Filename regression tests; PDF parsing is replaced with deterministic stubs."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def load_utils():
    project = Path(__file__).resolve().parents[1] / "project"
    pymupdf = ModuleType("pymupdf")
    pymupdf.__path__ = []
    pymupdf.open = Mock(side_effect=lambda path: SimpleNamespace(name=str(path)))
    converter = ModuleType("pymupdf4llm")
    converter.to_markdown = Mock(
        side_effect=lambda doc, **kwargs: f"Converted {Path(doc.name).name}"
    )
    config = ModuleType("config")
    dependencies = {
        "pymupdf": pymupdf,
        "pymupdf.layout": ModuleType("pymupdf.layout"),
        "pymupdf4llm": converter,
        "tiktoken": ModuleType("tiktoken"),
        "config": config,
    }
    spec = importlib.util.spec_from_file_location("filename_test_utils", project / "utils.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies), patch.dict("os.environ"):
        spec.loader.exec_module(module)
    return module


class PdfFilenameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "markdown"
        self.output.mkdir()
        self.utils = load_utils()
        self.utils.config.MARKDOWN_DIR = str(self.output)

    def test_conversion_preserves_the_complete_stem(self):
        for name in ("guide.pdf", "guide.v1.pdf", "guide.v2.pdf", "guide.en.v1.pdf"):
            with self.subTest(name=name):
                self.utils.pdf_to_markdown(self.root / name, self.output)
                target = self.output / Path(name).with_suffix(".md")
                self.assertTrue(target.is_file(), target.name)
                self.assertEqual(target.read_text(encoding="utf-8"), f"Converted {name}")

    def test_batch_converts_both_versions_without_a_collision(self):
        for name in ("guide.v1.pdf", "guide.v2.pdf"):
            (self.root / name).touch()
        self.utils.pdfs_to_markdowns(str(self.root / "*.pdf"))
        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()),
            ["guide.v1.md", "guide.v2.md"],
        )
        for version in ("v1", "v2"):
            self.assertEqual(
                (self.output / f"guide.{version}.md").read_text(encoding="utf-8"),
                f"Converted guide.{version}.pdf",
            )

    def test_existing_matching_markdown_is_skipped(self):
        (self.root / "guide.v1.pdf").touch()
        target = self.output / "guide.v1.md"
        target.write_text("Existing conversion", encoding="utf-8")
        self.utils.pdfs_to_markdowns(str(self.root / "*.pdf"))
        self.utils.pymupdf.open.assert_not_called()
        self.assertEqual(target.read_text(encoding="utf-8"), "Existing conversion")

    def test_overwrite_replaces_only_the_matching_version(self):
        (self.root / "guide.v1.pdf").touch()
        target = self.output / "guide.v1.md"
        other = self.output / "guide.v2.md"
        target.write_text("Old conversion", encoding="utf-8")
        other.write_text("Keep this version", encoding="utf-8")
        self.utils.pdfs_to_markdowns(str(self.root / "*.pdf"), overwrite=True)
        self.assertEqual(target.read_text(encoding="utf-8"), "Converted guide.v1.pdf")
        self.assertEqual(other.read_text(encoding="utf-8"), "Keep this version")


if __name__ == "__main__":
    unittest.main()
