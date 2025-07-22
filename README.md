# PLSCAN

Persistent Leaf Spatial Clustering for Applications with Noise


```bash
pip install --no-deps --no-build-isolation --config-settings=cmake.build-type=Debug -ve .
```

On windows run build commands in the developer powershell configured for an
amd64 target architecture!
```powershell
pip install --no-deps --no-build-isolation --config-settings=cmake.build-type=Debug --config-settings=cmake.args="-T ClangCL" -ve . 
```