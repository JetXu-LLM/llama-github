# Installation

This document provides instructions on how to install the `llama-github` library in your Python environment.

## Prerequisites

Before installing `llama-github`, ensure that you have the following prerequisites:

- Python 3.6 or above installed on your system.
- `pip` package installer for Python.

## Installation Methods

There are two main methods to install `llama-github`:

1. Installing from PyPI (recommended)
2. Installing from the GitHub repository

### Installing from PyPI

The easiest and recommended way to install `llama-github` is using `pip`, the Python package installer. To install `llama-github` from PyPI, run the following command:

```bash
pip install llama-github
```

This command will download and install the latest stable release of `llama-github` along with its dependencies.

### Installing from the GitHub Repository

If you want to install the latest development version of `llama-github` or contribute to the project, you can install it directly from the GitHub repository. Follow these steps:

1. Clone the `llama-github` repository from GitHub:

   ```bash
   git clone https://github.com/JetXu-LLM/llama-github.git
   ```

2. Navigate to the cloned repository directory:

   ```bash
   cd llama-github
   ```

3. Install the library using `pip`:

   ```bash
   pip install .
   ```

   This command will install `llama-github` along with its dependencies.

## Development Installation

If you are a developer contributing to the `llama-github` project or want to use the development version, you can install the library with additional development dependencies. Use the following command:

```bash
pip install -e .[dev]
```

This command installs `llama-github` in editable mode (`-e`) along with the development dependencies specified in the `dev` extras require section of the `setup.cfg` file.

## Verifying the Installation

To verify that `llama-github` is installed correctly, you can run the following command:

```bash
python -c "import llama_github; print(llama_github.__version__)"
```

If the installation is successful, it will print the version number of `llama-github`.

## Updating the Library

To update `llama-github` to the latest version, you can use the following command:

```bash
pip install --upgrade llama-github
```

This command will upgrade `llama-github` to the latest version available on PyPI.

## Uninstalling the Library

If you want to uninstall `llama-github` from your Python environment, you can use the following command:

```bash
pip uninstall llama-github
```

This command will remove the `llama-github` library from your system.

## Conclusion

You have now successfully installed `llama-github` in your Python environment. You can start using the library by importing it in your Python scripts or interactive sessions.

If you encounter any issues during the installation process or have any questions, please refer to the project's [documentation](usage.md) or open an issue on the [GitHub repository](https://github.com/JetXu-LLM/llama-github/issues).

Happy coding with `llama-github`!