Local development
=================

We recommend `uv <https://docs.astral.sh/uv/>`_ for building the project. A
C++23 compiler with OpenMP support is required on your system. Platform-specific
setup is described below.

Once the prerequisites are in place, set up the development environment for the
first time with:

.. code-block:: bash

  uv sync --only-dev
  uv pip install --no-build-isolation -ve . \
    --config-settings cmake.build-type=Debug

The first command creates the ``.venv`` and installs all dependencies. The
second compiles and installs the ``fast_plscan`` package without build
isolation. Running `uv sync` or `uv run` will revert the ``fast_plscan`` package
to the default state (isolated build, Release mode).

Repeat only the second command whenever C++ source files change. Python-only
changes are reflected immediately without any reinstall.

Linux
-----

It may be necessary to tell cmake which compiler it should use. For example,
using ``g++-14`` when that is not the system default:

.. code-block:: bash

  uv pip install --no-build-isolation -ve . \
    --config-settings cmake.args="-DCMAKE_CXX_COMPILER=g++-14"

MacOS
-----

MacOS requires installing OpenMP using homebrew:

.. code-block:: bash

  brew install libomp

Then either export ``OpenMP_ROOT`` in your shell profile:

.. code-block:: bash

  export OpenMP_ROOT=$(brew --prefix)/opt/libomp

or pass it as a cmake argument:

.. code-block:: bash

  uv pip install --no-build-isolation -ve . \
    --config-settings cmake.args="-DOpenMP_ROOT=$(brew --prefix)/opt/libomp"

Windows
-------

The default powershell terminal on windows is not configured for cmake to find
the correct OpenMP version. Instead, use a developer powershell configured for a
64-bit target architecture. To open such a terminal, run the following code in a
normal Powershell terminal:

.. code-block:: powershell

  $vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
  $iloc = & $vswhere -products * -latest -property installationpath
  $devddl = "$iloc/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
  Import-Module $devddl; Enter-VsDevShell -Arch amd64 -VsInstallPath $iloc -SkipAutomaticLocation

You may also need to install the `visual studio build tools
<https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`_.

Testing
-------

To enable C++ coverage instrumentation, build with:

.. code-block:: bash

  # Linux / macOS
  uv pip install --no-build-isolation -ve . \
    --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug"

  # Windows — -T ClangCL selects the LLVM toolset in the Visual Studio generator
  uv pip install --no-build-isolation -ve . \
    --config-settings cmake.args="-DPLSCAN_COVERAGE=ON;-DCMAKE_BUILD_TYPE=Debug;-T ClangCL"

Then collect coverage with:

.. code-block:: bash

  # Linux / macOS
  bash scripts/collect_coverage.sh

  # Windows (PowerShell)
  pwsh scripts\\collect_coverage.ps1

Run the normal tests with:

.. code-block:: bash

  pytest .
