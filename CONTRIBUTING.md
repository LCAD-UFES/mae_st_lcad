# Contributing to mae_st_lcad

How changes are made in this repository. [DESIGN.md](DESIGN.md) says what `main` changes over upstream; the tooling's own [CONTRIBUTING](https://github.com/LCAD-UFES/mae_st_lcad/blob/mae_st_tooling/CONTRIBUTING.md) covers its internals.

## Where a change belongs

- **`main`**: the model and data code. Domain changes (what range images need) and fixes that keep upstream running belong here, and so does efficiency work on the model. A change here must keep upstream's behaviour as the default unless the domain forbids it (as `spatial_resample=False` does), and say so in the commit message.
- **`mae_st_tooling`**: everything about *using* the model (configs, drivers, visualization, the installer). It imports `main` and never patches it; if the tooling needs something from `main`, that is a commit on `main`.

## Tests

Run both suites before committing; they run on CPU in seconds.

```bash
cd ~/mae_st_lcad   && python -m unittest discover -s tests -t .
cd ~/mae_st_tooling && python -m unittest discover -s tests -t .
```

A change of behaviour comes with a test. A change meant to preserve behaviour (a refactor, an efficiency change) comes with a check of numerical equivalence against the previous code, described in the commit message.

## Commits

- English, one logical change per commit.
- The message says what changed, **why**, and how it was checked. Upstream commits carry no explanations; ours are the documentation of the fork.
- No names of people or machines in code, docs or messages. Paths come from arguments or environment variables with defaults.

## Branches and tags

The repository has two branches, `main` and `mae_st_tooling`. Work is not kept in other long-lived branches. Superseded states are frozen as annotated tags `archive/<branch>-<yyyy-mm>`, so they stay reachable without appearing in the branch list:

```bash
git tag -l 'archive/*'
git switch -c <name> archive/<tag>     # to look at or continue from one
```

## Upstream

`main` keeps upstream's history (`facebookresearch/mae_st` up to `1e6b68a`) and its files: [PRETRAIN.md](PRETRAIN.md), [FINETUNE.md](FINETUNE.md), [DATASET.md](DATASET.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [LICENSE](LICENSE) (CC-BY-NC 4.0) are upstream's and are left as they are. Contributions are licensed under the same LICENSE.
