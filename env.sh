# Sourced by the launchers: activates the conda environment built by the installer.
#   MAE_ST_CONDA_ENV  environment name (default mae_st, the installer's)
#   CONDA_EXE         conda executable; set by `conda init`, else `conda` must be on PATH
_conda_base="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
[ -n "$_conda_base" ] || _conda_base="$(conda info --base 2>/dev/null)"
if [ -z "$_conda_base" ]; then echo "conda not found: set CONDA_EXE or put conda on PATH" >&2; return 1; fi
source "$_conda_base/etc/profile.d/conda.sh"
conda activate "${MAE_ST_CONDA_ENV:-mae_st}" || return 1
unset _conda_base
