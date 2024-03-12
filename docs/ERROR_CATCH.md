
# Error Catching

I have recorded the errors I encountered during the installation. If you have any questions, please feel free to open an issue.

- PyRender error.
```
# please add following to bashrc:
export DISPLAY=:0
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
```

- libGL error: failed to load driver: swrast
```
conda install -c conda-forge gcc
```

- torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGKILL
```
pip install h5py==3.3.0
```

- PyYAML (>=5.1.*)
```
pip install setuptools==61.1.0
```

- ERROR: Could not build wheels for mask2former, which is required to install pyproject.toml-based projects
```
See https://github.com/NVlabs/ODISE/issues/19#issuecomment-1592580278
```

- wandb 'run = wi.init()' error
```
pip install wandb==0.14.0
```

- ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' 
```
pip install torchmetrics==0.6.0
```

You can also refer to [GNFactor's error catching](https://github.com/YanjieZe/GNFactor/blob/main/docs/ERROR_CATCH.md) for more error types.
