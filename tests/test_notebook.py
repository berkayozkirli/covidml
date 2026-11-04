"""Check the moved notebook's package imports and checkout-relative paths."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from covidml.paths import find_project_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_project_root_from_nested_directory(self):
        self.assertEqual(find_project_root(PROJECT_ROOT / "notebooks"), PROJECT_ROOT)
        self.assertEqual(find_project_root(PROJECT_ROOT), PROJECT_ROOT)

    def test_missing_checkout_has_a_useful_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "project root"):
                find_project_root(directory)

    def test_notebook_code_compiles_and_imports_reusable_definitions(self):
        notebook = json.loads((PROJECT_ROOT / "notebooks/project_notebook.ipynb").read_text())
        imports = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            tree = ast.parse(code)
            compile(tree, "project_notebook.ipynb", "exec")
            self.assertFalse(
                any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body),
                "Reusable definitions belong in the package",
            )
            imports.extend(
                node for node in tree.body
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("covidml")
            )
        exec(compile(ast.Module(body=imports, type_ignores=[]), "notebook imports", "exec"), {})

    def test_package_imports_from_notebooks_directory(self):
        script = (
            "from covidml.models import TCN, make_xgboost_pipeline; "
            "from covidml.data import COVIDSequenceDataset, make_sequence_loader; "
            "from covidml.training import train_tcn, predict_tcn; "
            "from covidml.paths import find_project_root; "
            "root = find_project_root(); "
            "assert (root / 'src/covidml').is_dir(); "
            "assert (root / 'notebooks/project_notebook.ipynb').is_file()"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT / "notebooks",
            check=True,
            capture_output=True,
            text=True,
        )
