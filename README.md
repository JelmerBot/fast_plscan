# PLSCAN

Persistent Leaf Spatial Clustering for Applications with Noise


## TODO

- [x] ci/cd pipeline.
  - [x] test cibuild wheels config locally (linux passes)
  - [x] push workflow
  - [x] test cibuild wheels action
    - [x] ubuntu
    - [x] windows
    - [x] macos
    - [x] macos-arm
    - [ ] windows-arm (only available on public repos)
    - [ ] ubuntu-arm (only available on public repos)
- [x] update branch protection rule
- [ ] setup docs config
- [ ] support missing values in feature vectors.
- [ ] compute core distances based on sample weights.
- [ ] support core graph / sub-cluster clustering.
- [ ] implement prediction after fit functions?

## Development

The development workflow works best by pre-installing python dependencies with
`pip` (or alternatives):

```bash
pip install numpy scipy matplotlib scikit-learn scikit-build-core nanobind setuptools_scm
```

Building the package requires a C++ 20 compiler with OpenMP support. The OpenMP
version must support user-defined reductions. Selecting the proper OpenMP
version requires some additional configuration, see below. Assuming the compiler
and OpenMP are present, the package can be compiled and installed with:

```bash
pip install --no-deps --no-build-isolation -ve .
```

To change the build type, add `-C cmake.build-type=Debug` or `-C
cmake.build-type=Release` to the command.

`scikit-build-core` also experimentally editable installs (see [their
documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs)):

```bash
pip install --no-deps --no-build-isolation -C editable.rebuild=true -ve .
```

### Linux

It may be necessary to tell cmake which compiler it should use. For example,
using `g++-14` when that is not the system default can be done by adding a `-C
cmake.args="-DCMAKE_CXX_COMPILER=g++-14"` option. The `-C cmake.args=...` option
does not have to be repeated on rebuilds.

### MacOS

MacOS requires installing OpenMP using homebrew:

```bash
brew install libomp
```

Also update the `~/.zshrc` config file with:

```bash
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```

### Windows

The default MSVC C++ compiler on windows does not support a recent enough
OpenMP. In addition, the default powershell terminal on windows is not
configured for cmake to find the correct OpenMP version. Instead, use a
developer powershell configured for a 64-bit target architecture. To open such a
terminal, run the following code in a normal Powershell terminal:

```powershell
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$iloc = & $vswhere -products * -latest -property installationpath
$devddl = "$iloc/Common7/Tools/Microsoft.VisualStudio.DevShell.dll"
Import-Module $devddl; Enter-VsDevShell -Arch amd64 -VsInstallPath $iloc -SkipAutomaticLocation
```

In addition, select the MSVC Clang compiler using `-C cmake.args="-T ClangCL"`
the first time the package is installed:

```powershell
pip install --no-deps --no-build-isolation -C cmake.args="-T ClangCL" -ve . 
```

The `-C cmake.args=...` option does not have to be repeated on rebuilds.