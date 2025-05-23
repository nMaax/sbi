# How to contribute

!!! important

    By participating in the `sbi` community, all members are expected to comply with
    our [Code of Conduct](code_of_conduct.md). This ensures a positive and inclusive
    environment for everyone involved.

## User experiences, bugs, and feature requests

If you are using `sbi` to infer the parameters of a simulator, we would be
delighted to know how it worked for you. If it didn't work according to plan,
please open up an [issue](https://github.com/sbi-dev/sbi/issues) or
[discussion](https://github.com/sbi-dev/sbi/discussions) and tell us more about
your use case: the dimensionality of the input parameters and of the output,
as well as the setup you used to run inference (i.e., number of simulations,
number of rounds, etc.).

To report bugs and suggest features -- including better documentation --
please equally head over to [issues on GitHub](https://github.com/sbi-dev/sbi/issues)
and tell us everything.

## Contributing code

Contributions to the `sbi` package are always welcome! The preferred way to do
it is via pull requests onto our [main repository](https://github.com/sbi-dev/sbi).
To give credit to contributors, we consider adding contributors who repeatedly
and substantially contributed to `sbi` to the list of authors of the package at
the end of every year. Additionally, we mention all contributors in the releases.

!!! note
    To avoid doing duplicated work, we strongly suggest that you go take
    a look at our current [open issues](https://github.com/sbi-dev/sbi/issues) and
    [pull requests](https://github.com/sbi-dev/sbi/pulls) to see if someone else is
    already doing it. Also, in case you're planning to work on something that has not
    yet been proposed by others (e.g. adding a new feature, adding a new example),
    it is preferable to first open a new issue explaining what you intend to
    propose and then working on your pull request after getting some feedback from
    others.

### Contribution workflow

The following steps describe all parts of the workflow for doing a contribution
such as installing locally `sbi` from source, creating a `conda` environment,
setting up your `git` repository, etc. We've taken strong inspiration from the
contribution guides of
[`scikit-learn`](https://scikit-learn.org/stable/developers/contributing.html)
and [`mne`](https://mne.tools/stable/development/contributing.html):

**Step 1**: [Create an account](https://github.com/) on GitHub if you do not
already have one.

**Step 2**: Fork the [project repository](https://github.com/sbi-dev/sbi): click
on the ‘Fork’ button near the top of the page. This will create a copy of the
`sbi` codebase under your GitHub user account. See more details on how to fork
a repository [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

**Step 3**: Clone your fork of the `sbi` repo from your GitHub account to your
local disk:

```bash
git clone git@github.com:$USERNAME/sbi.git
cd sbi
```

**Step 4**: Install a recent version of Python (we currently recommend 3.10)
for instance using [`miniforge`](https://github.com/conda-forge/miniforge). We
strongly recommend you create a specific `conda` environment for doing
development on `sbi` as per:

```bash
conda create -n sbi_dev python=3.10
conda activate sbi_dev
```
If you are using [`uv`](http://docs.astral.sh/uv/), set up a virtual environment by running:

```bash
uv venv
```

Then activate the virtual enviroment by running:

- For `macOS` or `Linux` users
  ```bash
  source .venv/bin/activate
  ```

- For `Windows` users
  ```bash
  .venv\Scripts\activate
  ```

**Step 5**: Install `sbi` in editable mode with

```bash
pip install -e ".[dev]"
```

This installs the `sbi` package into the current environment by creating a
link to the source code directory (instead of copying the code to pip’s `site_packages`
directory, which is what normally happens). This means that any edits you make
to the `sbi` source code will be reflected the next time you open a Python interpreter
and `import sbi` (the `-e` flag of pip stands for an “editable” installation,
and the `dev` flag installs development and testing dependencies). This requires
at least Python 3.8.

If you are using [`uv`](http://docs.astral.sh/uv/) you can install the development dependencies using:

```bash
uv sync --extra dev
```

**Step 6**: Add the upstream remote. This saves a reference to the main `sbi`
repository, which you can use to keep your repository synchronized with the latest
changes:

```bash
git remote add upstream git@github.com:sbi-dev/sbi.git
```

Check that the upstream and origin remote aliases are configured correctly by
running `git remote -v` which should display:

```bash
origin  git@github.com:$USERNAME/sbi.git (fetch)
origin  git@github.com:$USERNAME/sbi.git (push)
upstream        git@github.com:sbi-dev/sbi.git (fetch)
upstream        git@github.com:sbi-dev/sbi.git (push)
```

**Step 7**: Install `pre-commit` to run code style checks before each commit:

```bash
pip install pre-commit
pre-commit install
```

If you are using [`uv`](http://docs.astral.sh/uv/) install `pre-commit` using:
```bash
pre-commit install
```

You should now have a working installation of `sbi` and a git repository
properly configured for making contributions. The following steps describe the
process of modifying code and submitting a pull request:

**Step 8**: Synchronize your main branch with the upstream/main branch. See more
details on [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork):

```bash
git checkout main
git fetch upstream
git merge upstream/main
```

**Step 9**: Create a feature branch to hold your development changes:

```bash
git checkout -b my_feature
```

and start making changes. Always use a feature branch! It’s good practice
to never work on the main branch, as this allows you to easily get back to a
working state of the code if needed (e.g., if you’re working on multiple
changes at once, or need to pull in recent changes from someone else to get
your new feature to work properly). In most cases you should make PRs into the
upstream’s main branch.

**Step 10**: Develop your code on your feature branch on the computer, using
Git to do the version control. When you’re done editing, add changed files
using `git add` and then `git commit` to record your changes:

```bash
git add modified_files
git commit -m "description of your commit"
```

Then push the changes to your GitHub account with:

```bash
git push -u origin my_feature
```

The `-u` flag ensures that your local branch will be automatically linked with
the remote branch, so you can later use `git push` and `git pull` without any
extra arguments.

**Step 11**: Follow [these](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
instructions to create a pull request from your fork.
This will send a notification to `sbi` maintainers and trigger reviews and comments
regarding your contribution.

!!! note
    It is often helpful to keep your local feature branch synchronized
    with the latest changes of the main `sbi` repository:
    ```
    git fetch upstream
    git merge upstream/main
    ```

### Style conventions and testing

All our docstrings and comments are written following the [Google
Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

For code linting and formating, we use [`ruff`](https://docs.astral.sh/ruff/),
which is installed alongside `sbi`.

You can exclude slow tests and those which require a GPU with

```bash
pytest -m "not slow and not gpu"
```

Additionally, we recommend to run tests with

```bash
pytest -n auto -m "not slow and not gpu"
```

in parallel. GPU tests should probably not be run this way. If you see unexpected
behavior (tests fail if they shouldn't), try to run them without `-n auto` and
see if it persists. When writing new tests and debugging things, it may make sense
to also run them without `-n auto`.

When you create a PR onto `main`, our Continuous Integration (CI) actions on
GitHub will perform the following checks:

- **[`ruff`](https://docs.astral.sh/ruff/formatter/)** for linting and formatting
  (including `black`, `isort`, and `flake8`)
- **[`pyright`](https://github.com/Microsoft/pyright)** for static type checking.
- **[`pytest`](https://docs.pytest.org/en/stable/index.html)** for running a subset of
  fast tests from our test suite.

If any of these fail, try reproducing and solving the error locally:

- **`ruff`**: Make sure you have `pre-commit` installed locally with the same version as
 specified in the
 [`pyproject.toml`](https://github.com/sbi-dev/sbi/blob/main/pyproject.toml). Execute it
  using `pre-commit run --all-files`. `ruff` tends to give informative error messages
  that help you fix the problem. Note that pre-commit only detects problems with `ruff`
  linting and formatting, but does not fix them. You can fix them either by running
  `ruff check . --fix(linting)`, followed by `ruff format . --fix(formatting)`, or by
  hand.
- **`pyright`**: Run it locally using `pyright sbi/` and ensure you are using
the same
  `pyright` version as used in the CI (which is the case if you have installed
  it with `pip install -e ".[dev]"` but note that you have to rerun it once
  someone updates the version in the `pyproject.toml`).
  - Known issues and fixes:
    - If using `**kwargs`, you either have to specify all possible types of
    `kwargs`, e.g. `**kwargs: Union[int, boolean]` or use `**kwargs: Any`
- **`pytest`**: On GitHub Actions you can see which test failed. Reproduce it
locally, e.g., using `pytest -n auto tests/linearGaussian_snpe_test.py`. Note
that this will run for a few minutes and should result in passes and expected
fails (xfailed).
- Commit and push again until CI tests pass. Don't hesitate to ask for help by
  commenting on the PR.

#### mini-sbibm tests

As SBI is a fundamentally data-driven approach, we are not only interested in whether
the modifications to the codebase "pass the tests" but also in whether they improve or
at least do not deteriorate the performance of the package for inference. To this end,
we have a set of *mini-sbibm* tests (a minimal version of the sbi benchmarking package [`sbibm`](https://github.com/sbi-benchmark/sbibm)) that are intended for developers to run locally.

These tests differ from the regular tests in that they always pass (provided there
are no errors) but output performance metrics that can be compared, e.g., to the
performance metrics of the main branch or relative to each other. The user-facing API
is available via `pytest` through custom flags. To run the mini-sbibm tests, you can use
the following command:

```bash
    pytest --bm
```

This will run all the mini-sbibm tests on all methods with default parameters and output
the performance metrics nicely formatted to the console. If you have multiple CPU cores
available, you can run the tests in parallel using the `-n auto` flag:

```bash
    pytest --bm -n auto
```

What if you are currently working on a specific method and you want to run the
mini-sbibm tests only for this class of methods? You can use the `--bm-mode` flag:

```bash
    pytest --bm --bm-mode nspe
```

This will run the mini-sbibm tests only for methods of the `nspe` class, but with a
few major hyperparameter choices, such as different base network architectures and
different diffusion processes.

The currently available modes are: `"npe"`, `"nle"`, `"nre"`, `"fmpe"`, `"npse"`,
`"snpe"`, `"snle"`, and `"snre"`. If you require another mode, you can add it to the
test suite in `tests/test_bm.py`.

## Contributing to the documentation

Most of the documentation for `sbi` is written in markdown and the website is generated
using `mkdocs` with `mkdocstrings` and `mike`. The tutorials and examples are converted
from jupyter notebooks into markdown files to be shown on the website. To work on
improvements of the documentation, you should first install the `doc` dependencies:

```bash
pip install -e ".[doc]"
```

If you are using [`uv`](http://docs.astral.sh/uv/) you can install the `doc` dependencies using:

```bash
uv sync --all-extras
```

Then, you can build the website locally by executing in the `mkdocs` folder

```bash
mkdocs serve
```

This will build the website on a local host address shown in the terminal. Changes to
the website files or a browser refresh will immediately rebuild the website.

If you updated the tutorials or examples, you need to convert them to markdown first:

```bash
cd docs
jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir docs/tutorials/
mkdocs serve
```

### Using AI Coding Assistants

We understand that AI coding assistants (like GitHub Copilot, ChatGPT, etc.) can be
helpful tools.  You are welcome to use them when contributing to this project, but *with
caution and responsibility*.

- **Understand the Code:** Do *not* blindly accept suggestions from AI assistants. You
    are responsible for ensuring that any code you submit (whether written by you or
    generated by an AI) is correct, efficient, secure, and follows all project
    guidelines.
- **Thoroughly Review AI-Generated Code:** Treat AI-generated code as you would any
  other code you didn't write yourself: review it carefully, line by line.  Understand
  what it does, how it does it, and why it's the best approach.
- **Test Extensively:** AI assistants can make mistakes (hallucinations, subtle bugs,
  inefficient code).  Write comprehensive unit tests to verify the correctness of any
  AI-generated code.  Don't rely solely on the AI to test its own code.
- **Attribution:** If a significant portion of the code was generated by an AI
  assistant, briefly mention this in your commit message or pull request description
  (e.g., "Implemented feature X with assistance from GitHub Copilot"). This helps
  reviewers understand the origin of the code.  However, you remain responsible for the
  code's correctness.
- **Maintainability:** Make sure the code is well-formatted, commented and does follow
  our code style.

**In essence: Use AI assistants as a *tool* to enhance your productivity, but *you* are
the programmer.  You are ultimately responsible for the quality and correctness of the
code you contribute.**
