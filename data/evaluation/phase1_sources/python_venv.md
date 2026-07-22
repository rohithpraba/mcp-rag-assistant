# Python Virtual Environment Facts

The venv module creates lightweight virtual environments. Each environment has an independent set of installed Python packages in its site-packages directory.

Creating an environment makes a target directory and writes a configuration file named pyvenv.cfg.

The environment contains a Python executable and activation scripts. On Windows, these are placed in the Scripts subdirectory. On POSIX systems, they are placed in the bin subdirectory.

A virtual environment is isolated from the base Python installation by default, although system packages can be exposed through configuration.
