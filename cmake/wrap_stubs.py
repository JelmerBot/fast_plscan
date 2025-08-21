"""
Wraps generated stubs in runtime modules so sphinx can properly document them.

See:
https://github.com/tox-dev/sphinx-autodoc-typehints/issues/161#issuecomment-1398781975
"""

import re
import sys
from textwrap import dedent
from pathlib import Path


def main(path, stub, docs_out, mod_out, project_name):
    # Read the stubs file
    in_file = Path(path) / stub
    with open(in_file) as f:
        stubs = f.read()

    # Find each definition
    ends = []
    indents = []
    matcher = re.compile(r"([ \t]*def )((?!(def|class) ).|\s)*(\"\"\"|\.\.\.)")
    for m in matcher.finditer(stubs):
        groups = m.groups()
        if groups[-1] == "...":
            continue  # Skip stubs that are already marked as not implemented
        ends.append(m.end())
        indents.append(len(groups[0]))

    # Insert not implemented annotation
    offset = 0
    for end, indent in zip(ends, indents):
        idx = end + offset
        new_line = "\n" + " " * indent + "..."
        stubs = stubs[:idx] + new_line + stubs[idx:]
        offset += len(new_line)

    # Correct imports
    # stubs = stubs.replace("import np", "import numpy as np")
    # child_modules = set(re.findall(rf"{project_name}\.(\w+)\.\w+", stubs))
    # stubs = re.sub(rf"{project_name}\.(\w+)\.(\w+)", r"\1.\2", stubs)
    # stubs = re.sub(
    #     rf"import {project_name}.*$",
    #     "from . import " + ", ".join(child_modules),
    #     stubs,
    #     flags=re.MULTILINE,
    # )

    # Write the modified stubs to the output file
    docs_file = Path(path) / docs_out
    with open(docs_file, "w") as f:
        f.write(stubs)

    # Write the wrapper module
    mods_file = Path(path) / mod_out
    docs_package = docs_out.replace(".py", "")
    impl_package = docs_package.replace("_docs", "")
    with open(mods_file, "w") as f:
        f.write(
            dedent(
                f"""\
                import builtins
                if not hasattr(builtins, "--BUILDING-DOCS--"):
                    from .{impl_package} import *
                else:
                    from .{docs_package} import *
                """
            )
        )


if __name__ == "__main__":
    main(*sys.argv[1:6])
