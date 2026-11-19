import ast
import copy
from pathlib import Path


def extract_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    found = {
        n.name: copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    for name in names:
        if name not in found:
            raise RuntimeError(f"Missing official function {name} in {path}")
        fn = found[name]
        fn.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace
