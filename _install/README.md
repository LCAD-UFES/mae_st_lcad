# Installing MAE-ST on a new machine

This directory contains an installer that builds the Python environment MAE-ST needs, and a **manifest** per machine describing exactly what to install and why.

Installing this repository by hand is a long trial-and-error session: the bundled `environment.yml` does not work, one dependency is frozen at a 2021 release, another is permanently broken and must be installed anyway, and PyTorch ships different binaries under the same version number depending on which CUDA runtime you want. The installer does not make those problems go away — it makes the answers reproducible and writes down the reasoning next to each one. `CLAUDE.md` in this directory holds the full design rationale; this file is the runbook.

## Quick start, on a machine that already has a manifest

```bash
./conda.sh <manifest.tsv> --dry-run   # print every command, run nothing

# After everything looks good:
./conda.sh <manifest.tsv>             # menu: rename env, inspect manifest, install
```

**The manifest is a required argument and there is no default.** The script installs whatever a manifest tells it to; running it without one is like calling a function without its argument, and no manifest is more canonical than another because no machine is. Run `./conda.sh` with no arguments and it lists the manifests sitting beside it.

A bare filename is resolved next to the script, so `./conda.sh deps.myhost.tsv` works from any directory; anything containing a slash is used as given.

Other flags: `--env <name>` overrides the environment name, `--yes` skips the menu and installs directly, `--dry-run` executes nothing, `--resolve-only` asks pip whether the manifest's versions exist and fit together without downloading anything (step 10), `-h` prints usage.

The script needs `bash` (it uses `BASH_SOURCE` and arrays); `sh conda.sh` will not work.

**If the preamble says your host does not match the manifest target, stop and follow the recipe below.** Installing anyway may pull wheels for the wrong platform, or a torch build that cannot see your GPU.

---

# Recipe: producing a manifest for a new machine

You never edit `conda.sh`. You write a new `deps.<host>.tsv`. The steps below establish, with evidence, what each row of that file should say.

Work through them in order and keep the outputs — step 9 turns them into the file. Every command is read-only; nothing here modifies the machine.

## How it fits together

Three kinds of file live in this directory, and only one of them is something you write. **`conda.sh`** is the installer and contains no package names at all; you never edit it. **`deps.<host>.tsv`** is a *manifest*: one per machine, listing every package to install, where to get it from, and — in a `note` column beside each row — why that version and not another. It is the script's only input, which is why it is a required argument. **`lock.<host>.txt`** is not an input at all: it is written *out* at the end of a successful install, a plain `pip freeze` of the environment that just worked, including the transitive dependencies the manifest never names. No tool ever reads it back; **you** do, once, to copy exact version numbers into the manifest. After that it has no further use, which is why it is git-ignored. **`logs/`** holds one record per real run — every command in order with its outcome — and is likewise local, kept for diagnosing a run rather than for sharing.

Running the installer against a manifest goes: validate the whole file and abort on any malformed row; compare this machine against the manifest's declared target and warn on any divergence; create the conda environment from the `python` row; install the remaining rows in stage order, merging rows that share an installer and source into single commands so co-dependent packages resolve together; then write the log and the lock, and finally verify by importing torch and running real work on the GPU. The recipe below wraps that in a loop: a new manifest starts *loose*, with ranges and `latest` where you cannot yet justify an exact number, and the first successful install produces a lock whose versions you read back into it (step 12). At that point the manifest is pinned, reproducible, and carries its own reasoning — and the lock is disposable.

## Step 1 — Identify the machine

```bash
hostname; uname -m
. /etc/os-release && echo "$ID $VERSION_ID"
lscpu | grep -Ei '^architecture|model name'
free -h | head -2
df -h "$HOME" | tail -1
```

**What you conclude:** the `os=` and `arch=` fields of the `#!target` line, and whether you have room. Budget roughly **8 GB for the environment plus 10 GB of download caches**; the caches are reclaimable (step 11), the environment is not.

`arch` matters more than it looks. On `x86_64` almost every package has a prebuilt wheel. On `aarch64` (ARM: DGX Spark, Grace, Jetson, Ampere servers, Apple silicon under Linux) many do not, and some conda packages do not exist at all.

## Step 2 — Identify the GPU and the CUDA situation

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1
ls -d /usr/local/cuda* 2>/dev/null
command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "no nvcc in PATH"
```

**What you conclude:** `gpu=` is the reported name. `sm=` is `compute_cap` with the dot removed — capability `12.1` becomes `sm=121`. `cuda=` is **the version `nvidia-smi` prints in its header**, which is the newest CUDA runtime the *driver* can run — not the toolkit in `/usr/local/cuda*`. Those two numbers differ routinely and the script compares against the first one, so record that one.

You need `nvcc` only if some row in your manifest compiles CUDA code. For this repository nothing does: it is pure Python over torch, and the torch wheel ships its own CUDA runtime.

If `nvidia-smi` is missing or reports no device, you are building a CPU-only environment. Use `index=https://download.pytorch.org/whl/cpu` in step 6 and expect verification to warn that no GPU is visible, which is then correct rather than alarming.

## Step 3 — Locate conda

```bash
echo "CONDA_EXE=$CONDA_EXE"; command -v conda
ls -d ~/miniconda3 ~/anaconda3 ~/miniforge3 ~/mambaforge /opt/conda 2>/dev/null
conda --version && conda env list && conda config --show channels
```

**What you conclude:** whether conda exists at all, and which channels are configured. The installer searches eight prefixes, so it will find conda even if it is not on `PATH`. If nothing turns up, install **miniforge** and repeat this step to confirm it is now visible — nothing you have done so far is invalidated, since steps 1 and 2 describe the hardware and do not depend on conda.

Channels matter for one row only — the `python` row of stage `00-env`, which becomes `conda create`. `channel=conda-forge` is the default choice and works on every platform this repository targets. Add `;nodefaults` if you want conda-forge exclusively rather than merely first.

## Step 4 — Check that the package indexes are reachable

```bash
for u in https://pypi.org/simple/ \
         https://download.pytorch.org/whl/ \
         https://conda.anaconda.org/conda-forge/noarch/repodata.json ; do
  printf '%-64s ' "$u"
  curl -s -o /dev/null -w '%{http_code}\n' --max-time 20 "$u"
done
```

**What you conclude:** three `200`s mean the plain manifest will work. Anything else means this machine needs a mirror or a proxy, which is expressed in the `source` column — `index=` to replace the default index, `extra=` to add one, `links=` for a vendor page of `.whl` files that is not a proper index. Do that in the manifest; never by editing the script.

## Step 5 — Look for an environment that already works here

This step costs one command and can save the most time in the whole recipe: a machine that already runs torch on this GPU has already answered, by demonstration, the question that steps 6 and 7 otherwise settle through several multi-gigabyte downloads and a round of trial and error.

```bash
conda env list
# then, for each existing environment that might have torch:
<prefix>/bin/python - <<'PY'
import torch
print("torch      ", torch.__version__)
print("built cuda ", torch.version.cuda)
print("arch list  ", torch.cuda.get_arch_list())
print("available  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

**What you conclude:** if any environment on this machine already runs torch on this GPU, **that exact version and build is your strongest candidate**, because it is the only evidence you have that is not a guess. A version like `2.11.0+cu128` tells you both the version *and* the index (`cu128`) in one string.

This is not hypothetical. On one ARM host with a Blackwell GPU, an environment belonging to an unrelated project was already running `torch 2.11.0+cu128`. That single string fixed both the version and the index, and it also demonstrated that the GPU worked with a build whose architecture list did not literally name the chip — a question that reads as alarming until you know the rule below.

## Step 6 — Choose the torch index

```bash
curl -s https://download.pytorch.org/whl/ \
  | grep -oE 'href="(cu[0-9]+|cpu|rocm[0-9.]+)/"' | sed 's|href="||; s|/"||' | sort -u
```

Each directory is a separate build of torch against a different CUDA runtime. Pick the one whose CUDA version supports your GPU's compute capability — newer chips need newer CUDA — and no newer than what step 2 said your driver can run.

Then confirm the version you want actually exists there:

```bash
python -m pip index versions torch --index-url https://download.pytorch.org/whl/cuXXX
```

**What you conclude:** the `source` column for the torch rows, `index=https://download.pytorch.org/whl/cuXXX`, and the version to pin.

**Do not skip the index.** `torch==2.11.0` from PyPI and `torch==2.11.0+cu128` from `download.pytorch.org/whl/cu128` are different binaries built against different CUDA runtimes, so the `source` column is not optional.

**And do not guess the version string from the wheel filename.** Whether the `+cuXXX` suffix belongs in the specifier differs between indexes, and the filename does not tell you. On `cu128`, `torch==2.11.0+cu128` is what resolves. On `cu117` the wheel is named `torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl` and yet `torch==2.0.1+cu117` does **not** resolve — only plain `torch==2.0.1` does, selecting that wheel because it is the only 2.0.1 the index carries.

Settle it with one command instead of reasoning about it. Silence means it resolved:

```bash
python -m pip install --dry-run --ignore-installed \
    --index-url https://download.pytorch.org/whl/cuXXX 'torch==<version>'
```

Once the manifest exists, `./conda.sh <manifest> --resolve-only` (step 10) makes this check for every row at once.

### A trap worth understanding: the architecture list

After installing, `torch.cuda.get_arch_list()` reports the GPU architectures that binary was compiled for. It is tempting to check whether your GPU's own `sm_XY` string appears in that list. **That check is wrong and will condemn healthy installs.**

CUDA guarantees minor-version binary compatibility: code built for `sm_XY` runs on any device of the same major `X` whose minor is at least `Y`. A torch whose list stops at `sm_120` runs natively on an `sm_121` device. The installer's verification implements this properly and reports four distinct outcomes — native, binary-compatible, PTX-JIT-only, or genuinely uncovered — and then runs real GPU work, because a completed matrix multiply is evidence while a list of tags is only a clue.

## Step 7 — Choose the Python version

The ceiling is set by your **oldest pinned dependency**, not by preference. Two of this repository's dependencies are old enough to decide it on their own: `timm 0.4.5` and `pytorchvideo 0.1.5` both predate Python 3.11, so **3.10** is the newest interpreter that can work, whatever the machine. Pick the exact patch release your conda channel offers.

Verify your chosen torch has a wheel for that interpreter. The reliable way is to let pip resolve without installing, which requires an interpreter of the right version — so create a throwaway environment first:

```bash
conda create -n _probe -c conda-forge python=3.10 -y
~/…/envs/_probe/bin/python -m pip install --dry-run --ignore-installed \
    --index-url https://download.pytorch.org/whl/cuXXX torch==<version>
conda env remove -n _probe -y
```

**What you conclude:** the `version` field of the `00-env` python row. Remember that **changing this row changes every compiled wheel in the file** — a manifest is only valid for one interpreter version.

## Step 8 — Check platform wheels for the remaining packages

For anything you are unsure about on a non-x86 machine:

```bash
curl -s https://pypi.org/pypi/<package>/json | python -c "
import json,sys
d = json.load(sys.stdin); v = d['info']['version']
print(v)
for f in d['releases'][v]:
    print('  ', f['filename'])
"
```

**What you conclude:** a filename ending in your platform tag (for example `manylinux_2_28_aarch64`) means a prebuilt binary exists. A release with only a `.tar.gz` means pip will **compile from source**, so you need a toolchain and should expect it to be slow or to fail. A single `py3-none-any` wheel means the package is pure Python and carries no architecture risk at all.

For conda rows, the equivalent question is:

```bash
conda search -c conda-forge --platform linux-<arch> <package>
```

## Step 9 — Write the manifest

Copy whichever existing manifest is closest to your machine — same architecture first, then same GPU vendor — and edit that copy. Never start from an empty file: the rows are easy to retype and the notes are not, and the notes are the valuable part.

```bash
ls *.tsv                       # what is already here
cp deps.<nearest>.tsv deps.<your-host>.tsv
$EDITOR deps.<your-host>.tsv
```

Name it after the machine, not after a role. `deps.<host>.tsv` keeps the filename honest about the one thing the file is specific to.

Change, in this order:

1. **The `#!target` line**, from steps 1 and 2. This is what the preamble compares against, so getting it right is what makes the tool able to warn the next person.
2. **The `00-env` python row**, from step 7.
3. **The two `20-torch` rows** — version and `index=` — from steps 5 and 6.
4. **Anything step 8 flagged.** If a package has no wheel for this platform, either accept the source build, find a `links=` page that has one, or pin an older version that does.

### Which rows you have to think about

Rows fall into three groups, and knowing which is which is most of the work:

**Repository-dependent — identical on every machine.** `timm==0.4.5`, `iopath`, `simplejson`, `psutil`, `av`, `pillow`, `tensorboard`, and the Python ceiling. These follow from what the source code imports. They are the reason you copy a manifest instead of writing one, and they need no attention.

**Machine-dependent — the obvious ones.** `torch`, `torchvision`, and the index they come from. Steps 2, 5 and 6 decide these.

**Torch-generation-dependent — the ones that bite.** A few packages must agree with *the torch you just chose* rather than with the machine directly.

`numpy` is the important case. Torch wheels are compiled against a specific NumPy ABI, so a torch from before the NumPy 2 transition cannot run with NumPy 2 installed. **PyTorch does not declare this constraint**, so pip resolves the broken combination without a word and the failure surfaces only at runtime. `opencv-python-headless` is tied into the same knot from the other side: releases 4.12 and newer *require* `numpy>=2` and will pull it in behind your back, so an old torch needs an opencv older than that as well.

The `required` flag on `pytorchvideo` belongs to this group too. That package cannot be imported alongside torchvision 0.17 or newer, but works with earlier ones — so whether its failure is expected depends entirely on which torch you picked.

**The rule of thumb: if a row was pinned from somebody else's lock, you cannot vouch for it.** Relax it to a range or `latest` and let step 12 re-pin it from *your* lock. Carry an exact version across only where you can state the reason: `timm==0.4.5` has one that holds on every machine, while `matplotlib==3.10.9` is merely what one machine happened to resolve on one day.

The file is tab-separated with nine columns; the header row and the extensive comments explain each one.

If this machine needs a package the others do not, add a row. If it must not install one, that is a removal — see the manifest's own notes for the current mechanism.

## Step 10 — Validate without installing anything

Two checks, in this order. Neither downloads a package or creates an environment.

**First, read the commands the manifest produces:**

```bash
./conda.sh deps.<host>.tsv --dry-run --yes
```

The preamble's five comparison rows should all say `match`. Then read the printed commands and confirm each is what you intended, particularly the index URL on the torch line. The manifest is fully parsed and validated before anything runs, so a malformed row, an unknown source scheme or a missing value aborts here, naming the offending package.

**Then ask pip whether the manifest is satisfiable:**

```bash
./conda.sh deps.<host>.tsv --resolve-only
```

This resolves every pip stage for the manifest's target interpreter and platform, reporting `RESOLVES`, `FAILS` or `UNCHECKED` per group, and exits non-zero if any required row fails. It is what catches a version that does not exist on the index you chose, a `+cuXXX` suffix in the wrong form, and two rows whose constraints cannot both hold. Seconds, against several gigabytes discovered mid-install.

### What its answer is worth

This check replaces **NO** step in this recipe. Steps 6 to 8 are how you *decide* what to write; this is how you *check* what you wrote, and a checker cannot do a decider's job.

**What it is actually for.** A real install checks strictly more than this does, so the value is never "catches something the install would miss". It is *where in the process an error surfaces*. The installer works through stages in order, and the largest download is not in the last stage — so a mistake anywhere after `20-torch` is only reached once torch has already been fetched in full.

A concrete example. A manifest written for a modern GPU is copied onto a CUDA 11.7 machine. The `numpy` row is correctly relaxed to `numpy<2`, since a torch of that generation cannot run beside NumPy 2. The `opencv-python-headless` row is left at the version the other machine had. Those two cannot both hold: opencv 4.12 and newer *require* `numpy>=2`. Both sides declare their constraint, so pip states the problem flatly as `ResolutionImpossible`.

Both of those rows sit in stage `30-core`. In a real install the error therefore appears only after `20-torch` has completed, which on the CUDA 11.7 index means fetching a **1.8 GB** torch wheel first. `--resolve-only` reaches the identical verdict, with identical certainty, in seconds and without downloading a single package. Same error, same confidence, 1.8 GB apart.

It is a cheap filter placed before the expensive part, and it is neither sound nor complete. Both directions matter:

**A failure is a strong signal, not a proof.** To resolve for an interpreter it is not running on, the check forbids source builds. A real install allows them. So a package whose requested version ships only as a source distribution can be reported as failing while the actual install would build it and succeed. The clear-cut case — a package with no wheels at all, like `pytorchvideo` — is detected and reported as `UNCHECKED` rather than failed, but the mixed case is not distinguishable. Read the pip error printed underneath before concluding the manifest is wrong.

**A pass proves considerably less than it looks.** Five distinct classes of problem survive it:

1. **Undeclared constraints.** Resolution only checks what packages *declare*. Torch does not declare a NumPy bound, so the pairing that breaks an older torch is approved by every resolver in existence.
2. **Conflicts between stages.** Each stage group is resolved as its own independent question. A manifest asking for `numpy==1.26.4` in one stage and `numpy==2.2.6` in another passes cleanly, and the real install simply ends with whichever ran last.
3. **Conda rows are not checked at all**, including the `python` row.
4. **Nothing is built.** A source distribution that fails to compile — missing toolchain, missing headers — looks identical to one that would succeed.
5. **Nothing is imported, and no GPU is touched.** A package can install perfectly and fail on import; a torch can install perfectly and have no kernels for the chip.

**Run it on the target machine when you can.** If the manifest's architecture matches the host, pip's own tag set decides which wheels fit and the answer is as good as this check gets. If it does not, it falls back to a hand-written list of platform tags, where a missing tag is indistinguishable from a missing package. That mode announces itself in the header and downgrades failures to `UNRESOLVED`.

The install's own verification step — which imports torch and runs real work on the GPU — is what settles the questions this cannot reach.

## Step 11 — Install into a scratch-isolated test environment

Do not install under the final environment name on the first attempt, and do not let the download caches grow in shared locations while you are still experimenting.

```bash
SCRATCH="$HOME/mae_st_install_scratch"
mkdir -p "$SCRATCH"/{conda-pkgs,pip-cache,tmp} _install/logs

CONDA_PKGS_DIRS="$SCRATCH/conda-pkgs" \
PIP_CACHE_DIR="$SCRATCH/pip-cache" \
TMPDIR="$SCRATCH/tmp" \
./conda.sh deps.<host>.tsv --env mae_st_test --yes 2>&1 \
  | tee "_install/logs/console.$(date +%Y%m%d-%H%M%S).log"
```

Keep the scratch directory on the **same filesystem** as your conda environments: conda hardlinks packages out of its cache, so deleting the cache afterwards leaves the environment intact. On a different filesystem it copies instead, doubling the space used.

To undo everything:

```bash
rm -rf "$HOME/mae_st_install_scratch"
conda env remove -n mae_st_test -y
```

`--yes` skips the menu **and** the confirmation before deleting an existing environment. On a re-run, drop it.

Each real run writes `logs/install.<tag>.<stamp>.log` — every command in order with its outcome, the manifest's checksum, and the resulting `conda list`.

## Step 12 — Pin what worked

The run also writes `lock.<host>.txt`, a `pip freeze` of the environment that just succeeded. The manifest is the **request** — what you asked for, and why. The lock is the **result** — what the resolver actually produced, transitive dependencies included. The lock has exactly one consumer: **a person reading versions out of it**. It is never passed to any tool, and running `pip install -r lock.<host>.txt` would defeat the whole design, because it would install without the index settings the manifest carries and quietly fetch a torch built for the wrong CUDA runtime.

So: open the lock, read the exact versions, and type them into the manifest's `version` column, replacing every `latest` and every open range. Editing by hand is the point rather than a limitation — it is where you notice things, and it is how the reason for a pin ends up written next to it in the `note` column. Fix what is known to work, and test upgrades deliberately somewhere else rather than discovering them on the machine people depend on.

Two things to know about how far this gets you. Pinning torch closes more than it seems, because torch declares its own CUDA stack with exact `==` constraints, so cuDNN, NCCL and friends follow automatically. But roughly twenty pure-Python transitive dependencies remain unpinned, and **their versions are not recorded anywhere once you are done** — the lock is discarded (below), so there is no answer to "which `protobuf` was running last August". The manifest lists them under `WHAT THIS FILE DOES NOT PIN` rather than implying a completeness it does not have. If that gap ever matters, the fix is a constraints file, not a preserved lock; see `CLAUDE.md`.

**The lock is not committed, and `.gitignore` drops it.** Once you have read its versions into the manifest and written down why each one is there, the manifest *contains* the lock — plus the reasoning, which the lock never had. Keeping both would leave two files asserting the same versions and ageing at different rates, and the one without explanations would be the one that looks authoritative. The lock is also reproducible: re-running the installer against a pinned manifest regenerates it.

If you deliberately want to preserve one — archiving a machine that is being decommissioned, say — move it into a subdirectory. The ignore rule is anchored to the installer directory, so a lock anywhere else is tracked normally. That is an escape hatch on purpose, and using it should be a decision you can point at.

## Step 13 — Acceptance test

Package imports are not proof. Run something real:

```bash
cd <repo root>
python visualize_video.py \
  --video_path demo/qZ_lFjCiR1c_000104_000114.avi \
  --checkpoint <path>/video-mae-200x4-nonorm.pth \
  --output_path <somewhere outside the repo>/recon.mp4 \
  --mask_ratio 0.95
```

The checkpoint is 3.8 GB from `https://dl.fbaipublicfiles.com/video-mae-200x4-nonorm.pth`. Put it outside the clone.

**What success looks like:** `Model load status: <All keys matched successfully>`, then a side-by-side video of original, masked and reconstructed frames in which the reconstruction is clearly recognisable. One command exercises torch and CUDA, OpenCV, PyAV demux and mux, and the model itself. On a GB10 it takes about fourteen seconds.

**What this does not prove:** the demo never imports the dataset loader, so it says nothing about whether *training* runs. That is a separate matter with separate problems.

Once this passes, reinstall under the real environment name (or clone the test one) and remove the scratch directory.

---

# Troubleshooting

**`conda not found`, but conda is installed.** Export `CONDA_EXE=/path/to/conda` and re-run. The script searches `PATH`, `$CONDA_EXE` and six common prefixes.

**`conda activate did not take effect`, and the script aborts.** This abort is deliberate and is protecting you. Without it, the entire dependency set installs into `base`. First check whether this shell was ever initialised for conda: if not, run `conda init bash`, open a new shell, and retry.

**A required install fails and the script stops.** Read the pip output above the abort rather than the abort itself. The common causes are a version that does not exist on the index you chose (step 6), and no wheel for your platform so pip tried to compile (step 8).

**`pytorchvideo` fails and the script continues with a yellow warning.** Expected on every machine. It is marked `required=no` precisely so the failure is observed rather than hidden. See the manifest note.

**Verification says no CUDA device is visible.** Either you installed a `cpu` build, or you installed a CUDA build whose runtime is newer than the driver supports. Compare the index you chose in step 6 against the driver's maximum from step 2.

**Verification warns that nothing covers your `sm_XY`.** This one is real: the torch build you chose has no kernels for this chip and no PTX to fall back on. Choose a different index.

**Verification prints `PTX JIT only`.** It works, but every kernel compiles on first use. Fine for a smoke test, worth fixing before long runs.

**The preamble says `DIFFERS` on a row.** The manifest was written for a different machine. It also prints where to look for the right values — that is the short version of this recipe.

**A change to the manifest seems to have no effect.** Rows are grouped into a single command by `installer+source+flags+required`, so two rows differing only in `note` still install together. Use `--dry-run` to see the commands that will actually run.
