"""All the process that can be run using nox.

The nox run are build in isolated environment that will be stored in .nox. to force the venv update, remove the .nox/xxx folder.
"""

import nox

@nox.session(reuse_venv=True, name="ee-test-sepal")
def ee_test_sepal(session):
    """Run all the test using the environment varialbe of the running machine."""
    session.install("git+https://github.com/openforis/earthengine-api.git@v0.1.374#egg=earthengine-api&subdirectory=python")
    session.install(".[test]")
    test_files = session.posargs or ["tests"]
    session.run("pytest", "--color=yes", "--cov", "--cov-report=html", *test_files)


@nox.session(reuse_venv=True)
def lint(session):
    """Apply the pre-commits."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--a", *session.posargs)


@nox.session(reuse_venv=True)
def test(session):
    """Run all the test using the environment varialbe of the running machine."""
    session.install(".[test]")
    test_files = session.posargs or ["tests"]
    session.run("pytest", "--color=yes", "--cov", "--cov-report=html", *test_files)


@nox.session(reuse_venv=True)
def docs(session):
    """Build the documentation."""
    session.install(".[doc]")
    # patch version in nox instead of pyproject to avoid blocking conda releases
    session.install("git+https://github.com/jenshnielsen/sphinx.git@fix_9884")
    session.install("git+https://github.com/12rambau/deprecated.git@master")
    session.run("sphinx-apidoc", "-o", "docs/api", "sampling_handler")
    session.run(
        "sphinx-build",
        "-v",
        "-b",
        "html",
        "docs",
        "docs/_build/html",
        "-w",
        "warnings.txt",
    )
    session.run("python", "tests/check_warnings.py")


@nox.session(name="mypy", reuse_venv=True)
def mypy(session):
    """Run a mypy check of the lib."""
    session.install(".[dev]")
    test_files = session.posargs or ["sampling_handler"]
    session.run("mypy", *test_files)
