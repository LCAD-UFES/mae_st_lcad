#!/bin/bash
# =============================================================================
# conda.sh  (v1.2.0) -- MAE-ST environment installer
#
# Requires bash (uses BASH_SOURCE, arrays, [[ ]]). Not POSIX sh.
#
# This script contains NO package list. Everything installed is read from the
# manifest named on the command line. To support a new machine, write a new
# manifest -- never edit this file.
#
# The manifest is the script's input, so it is a REQUIRED positional argument.
# There is no default: no manifest is more canonical than any other, exactly
# as no machine is.
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST=""
ENV_NAME=""
NONINTERACTIVE=0
DRY_RUN=0
RESOLVE_ONLY=0

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") <manifest.tsv> [options]

  <manifest.tsv>   REQUIRED. The dependency manifest for this machine.
                   A path, or a bare filename resolved next to this script.

Options:
  --env <name>     environment name, overriding the manifest's #!envname
  --yes, -y        skip the menu and install directly
  --dry-run        print every command, execute nothing
  --resolve-only   ask pip whether the manifest's versions exist and are
                   mutually satisfiable, for the target python and platform.
                   Downloads nothing, creates nothing, installs nothing.
  -h, --help       this text

A real run writes logs/install.<tag>.<stamp>.log and lock.<tag>.txt.
See README.md for how to produce a manifest for a new machine.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)          ENV_NAME="$2"; shift 2 ;;
        --yes|-y)       NONINTERACTIVE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --resolve-only) RESOLVE_ONLY=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            [[ -z "${MANIFEST}" ]] \
                || { echo "Unexpected extra argument: $1" >&2; usage >&2; exit 2; }
            MANIFEST="$1"; shift ;;
    esac
done

# A bare filename is resolved next to the script, so that running it from
# anywhere with 'conda.sh deps.myhost.tsv' works.
if [[ -n "${MANIFEST}" && "${MANIFEST}" != */* && ! -f "${MANIFEST}" ]]; then
    MANIFEST="${SCRIPT_DIR}/${MANIFEST}"
fi

# --- Output helpers ----------------------------------------------------------
if [[ -t 1 ]]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
else
    R=""; G=""; Y=""; B=""; N=""
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s%s%s\n' "$G" "$*" "$N"; }
warn() { printf '%s%s%s\n' "$Y" "$*" "$N"; }
err()  { printf '%s%s%s\n' "$R" "$*" "$N" >&2; }
die()  { err "ERROR: $*"; exit 1; }
rule() { printf '%s\n' "-----------------------------------------------------------------------"; }

# Missing manifest is a usage error, not a mystery: say what is available.
list_manifests() {
    local f found=0
    for f in "${SCRIPT_DIR}"/*.tsv; do
        [[ -e "${f}" ]] || continue
        printf '    %s\n' "$(basename "${f}")"; found=1
    done
    [[ ${found} -eq 1 ]] || printf '    (none yet -- see README.md, "Recipe")\n'
}

if [[ -z "${MANIFEST}" ]]; then
    err "No manifest given. It is required: this script installs what a"
    err "manifest tells it to, and there is no default."
    err ""
    err "  Manifests available next to this script:"
    list_manifests >&2
    err ""
    err "  Usage: $(basename "${BASH_SOURCE[0]}") <manifest.tsv> [options]"
    err "  If none of them targets this machine, README.md has the recipe"
    err "  for writing one."
    exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
    err "ERROR: manifest not found: ${MANIFEST}"
    err ""
    err "  Manifests available next to this script:"
    list_manifests >&2
    exit 1
fi

# --- Run record --------------------------------------------------------------
# An installer that leaves no trace of what the resolver actually chose cannot
# support "pin what works": there is nothing to read the pins back from. Every
# real run writes two artifacts --
#   logs/install.<tag>.<stamp>.log   what was run, in order, and how it ended
#   lock.<tag>.txt                   pip freeze of the resulting environment
# The manifest carries the INTENT and the rationale; the lock carries the
# RESULT, transitive dependencies included. Neither replaces the other.
MANIFEST_TAG="$(basename "${MANIFEST}")"
MANIFEST_TAG="${MANIFEST_TAG%.tsv}"
MANIFEST_TAG="${MANIFEST_TAG#deps.}"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/install.${MANIFEST_TAG}.$(date +%Y%m%d-%H%M%S).log"
LOCK_FILE="${SCRIPT_DIR}/lock.${MANIFEST_TAG}.txt"

logf() {   # append one line to the run log; silent no-op during --dry-run
    [[ ${DRY_RUN} -eq 1 ]] && return 0
    mkdir -p "${LOG_DIR}"
    printf '%s\n' "$*" >>"${LOG_FILE}"
}

# --- Manifest parsing --------------------------------------------------------
# Metadata lines look like:  #!target<TAB>key=value<TAB>key=value...
meta_get() {   # meta_get <section> <key>
    awk -F'\t' -v sec="#!$1" -v key="$2" '
        $1 == sec { for (i = 2; i <= NF; i++) {
                        split($i, kv, "=")
                        if (kv[1] == key) { sub(/^[^=]*=/, "", $i); print $i; exit }
                    } }' "${MANIFEST}"
}

# Data rows: not comments, not the header, non-empty.
rows() {
    awk -F'\t' 'NF >= 9 && $1 !~ /^#/ && $1 != "stage" && $1 != "" { print }' "${MANIFEST}"
}

# Rows the stage loop installs: everything except the 00-env/python row, which
# is consumed by 'conda create' instead. Every other 00-env row (a toolkit, a
# compiler) is installed like any other -- the old version filtered the whole
# 00-env stage out of the loop and dropped those rows without a word.
install_rows() {
    rows | awk -F'\t' '!($1 == "00-env" && $2 == "python")'
}

TARGET_OS="$(meta_get target os)"
TARGET_ARCH="$(meta_get target arch)"
TARGET_SM="$(meta_get target sm)"
TARGET_CUDA="$(meta_get target cuda)"
TARGET_PY="$(meta_get target python)"
TARGET_GPU="$(meta_get target gpu)"
[[ -n "${ENV_NAME}" ]] || ENV_NAME="$(meta_get envname mae_st)"
[[ -n "${ENV_NAME}" ]] || ENV_NAME="mae_st"

# --- Host detection ----------------------------------------------------------
detect_host() {
    HOST_ARCH="$(uname -m)"
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        HOST_OS="$(. /etc/os-release && printf '%s %s' "${ID}" "${VERSION_ID}")"
    else
        HOST_OS="unknown"
    fi
    HOST_GPU="?"; HOST_SM="?"; HOST_CUDA="?"
    if command -v nvidia-smi >/dev/null 2>&1; then
        # 'first line only' is written as awk-with-END rather than 'head -1'.
        # head exits as soon as it has its line, the producer upstream takes
        # SIGPIPE, and 'set -o pipefail' turns that into exit 141 -- fatal, and
        # racy, so it would strike intermittently. A multi-GPU host makes it
        # likely, since nvidia-smi then prints one line per device.
        HOST_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                    | awk 'NR == 1 { v = $0 } END { print v }')"
        local cc
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | awk 'NR == 1 { v = $0 } END { print v }')"
        [[ -n "${cc}" ]] && HOST_SM="${cc//./}"
        HOST_CUDA="$(nvidia-smi 2>/dev/null \
                     | awk '!f && match($0, /CUDA Version: *[0-9.]+/) {
                                s = substr($0, RSTART, RLENGTH)
                                sub(/CUDA Version: */, "", s); v = s; f = 1
                            } END { print v }')"
    fi
    HOST_NAME="$(hostname 2>/dev/null || echo '?')"
}

DIVERGED=0
cmp_line() {   # cmp_line <label> <target> <host>
    local mark
    if [[ "$2" == "$3" ]]; then
        mark="${G}match${N}"
    else
        mark="${Y}DIFFERS${N}"; DIVERGED=1
    fi
    printf '  %-10s %-24s %-24s %s\n' "$1" "${2:--}" "${3:--}" "${mark}"
}

show_preamble() {
    detect_host
    rule
    say "${B}MAE-ST environment installer${N}"
    say "  manifest:    $(basename "${MANIFEST}") (${MANIFEST_ROWS} rows, validated)"
    say "  environment: ${B}${ENV_NAME}${N}"
    rule
    printf '  %-10s %-24s %-24s %s\n' "" "TARGET (manifest)" "THIS HOST (${HOST_NAME})" ""
    cmp_line "os"    "${TARGET_OS}"   "${HOST_OS}"
    cmp_line "arch"  "${TARGET_ARCH}" "${HOST_ARCH}"
    cmp_line "gpu"   "${TARGET_GPU}"  "${HOST_GPU}"
    cmp_line "sm"    "${TARGET_SM}"   "${HOST_SM}"
    cmp_line "cuda"  "${TARGET_CUDA}" "${HOST_CUDA}"
    printf '  %-10s %-24s %-24s\n' "python" "${TARGET_PY}" "(created by this script)"
    rule
    if [[ ${DIVERGED} -eq 1 ]]; then
        show_divergence_help
    else
        ok "  Host matches the manifest target. Option 3 should just work."
        rule
    fi
}

show_divergence_help() {
    warn "  This host does NOT match what the manifest was built for."
    say  "  Installing anyway may pull wheels for the wrong platform, or a"
    say  "  torch build that cannot see the GPU. Write a new manifest instead:"
    say  ""
    say  "    cp ${MANIFEST} ${SCRIPT_DIR}/deps.${HOST_NAME}.tsv"
    say  "    \$EDITOR ${SCRIPT_DIR}/deps.${HOST_NAME}.tsv"
    say  "    ./conda.sh deps.${HOST_NAME}.tsv"
    say  ""
    say  "  ${B}Where to find the right values:${N}"
    say  ""
    say  "  torch index URL for this GPU"
    say  "    Compute capability here is sm_${HOST_SM}; pick the CUDA build that"
    say  "    supports it. Available indexes are listed at"
    say  "      https://download.pytorch.org/whl/"
    say  "    (one directory per build: cu126, cu128, cu129, cpu, rocm...)."
    say  "    Confirm a candidate has a wheel for this platform before committing:"
    say  "      pip index versions torch --index-url https://download.pytorch.org/whl/cuXXX"
    say  "    After install, the decisive check is:"
    say  "      python -c 'import torch; print(torch.cuda.get_arch_list())'"
    say  "    sm_${HOST_SM} must appear there, or the GPU is unusable."
    say  ""
    say  "  whether a PyPI package has a wheel for this platform"
    say  "      curl -s https://pypi.org/pypi/<pkg>/json | python -c \\"
    say  "        \"import json,sys; d=json.load(sys.stdin); v=d['info']['version'];"
    say  "         print([f['filename'] for f in d['releases'][v]])\""
    say  "    No matching wheel means pip builds from sdist -- expect a compiler"
    say  "    and expect it to be slow or to fail."
    say  ""
    say  "  conda channel availability"
    say  "      conda search -c conda-forge --platform linux-${HOST_ARCH} <pkg>"
    say  ""
    say  "  One shortcut that does NOT work on ARM: conda's pytorch-cuda"
    say  "  package does not exist for linux-aarch64 at all. On ARM, torch"
    say  "  comes from the pip index, never from conda."
    rule
}

# --- Manifest pretty-printer (menu option 2) ---------------------------------
show_manifest() {
    rule
    say "${B}Dependency manifest: ${MANIFEST}${N}"
    rule
    awk -F'\t' -v b="${B}" -v n="${N}" -v y="${Y}" '
        NF >= 9 && $1 !~ /^#/ && $1 != "stage" && $1 != "" {
            if ($1 != laststage) {
                if (laststage != "") printf "\n"
                printf "%s[%s]%s\n", b, $1, n
                laststage = $1
            }
            req = ($8 == "yes") ? "" : y "  (optional)" n
            printf "  %-24s %-12s %-6s %-8s %s%s\n", $2, $3, $5, $4, $6, req
            if ($7 != "-") printf "  %-24s flags: %s\n", "", $7
            # wrap the note at ~66 chars, indented under the package
            nnote = split($9, w, " "); line = ""
            for (i = 1; i <= nnote; i++) {
                if (length(line) + length(w[i]) + 1 > 66) {
                    printf "  %-24s | %s\n", "", line; line = w[i]
                } else {
                    line = (line == "") ? w[i] : line " " w[i]
                }
            }
            if (line != "") printf "  %-24s | %s\n", "", line
        }' "${MANIFEST}"
    say ""
    rule
    say "${B}Notes and deliberate omissions${N} (verbatim from the manifest)"
    rule
    # Everything after the NOTES banner, with the leading '#' stripped.
    awk '/^#\t={10,}/ { on = 1 } on && /^#/ { sub(/^#\t?/, "  "); print }' "${MANIFEST}"
    rule
}

# --- Conda discovery ---------------------------------------------------------
# Never assume ~/anaconda3. Machines in this group carry miniconda, miniforge
# and system-wide /opt/conda installs, and hardcoding one prefix means the
# script silently uses the wrong interpreter or claims conda is missing.
find_conda() {
    local candidates=(
        "${CONDA_EXE}"
        "$(command -v conda 2>/dev/null || true)"
        "${HOME}/miniconda3/bin/conda"
        "${HOME}/anaconda3/bin/conda"
        "${HOME}/mambaforge/bin/conda"
        "${HOME}/miniforge3/bin/conda"
        "/opt/conda/bin/conda"
        "/usr/local/miniconda3/bin/conda"
    )
    local c
    for c in "${candidates[@]}"; do
        [[ -n "${c}" && -x "${c}" ]] && { printf '%s' "${c}"; return 0; }
    done
    return 1
}

# --- Installation ------------------------------------------------------------
# Build the pip spec for one row: 'timm' + '0.4.5' -> 'timm==0.4.5'
# --- Source grammar ----------------------------------------------------------
# The 'source' column is a ';'-separated list of scheme=value pairs.
#
#   pip:    pypi | index=<url> | extra=<url> | links=<url>
#           direct=<url> | git=<url>@<ref> | local=<path>
#   conda:  channel=<name-or-url>[,<name-or-url>...] | nodefaults
#
# index/extra/links/channel/nodefaults become installer FLAGS.
# direct/git/local replace the package SPEC (they name the artifact outright).

split_source() {   # split_source <source> -> one 'scheme=value' per line
    local IFS=';' part
    for part in $1; do
        [[ -n "${part}" ]] && printf '%s\n' "${part}"
    done
}

source_flags() {   # source_flags <installer> <source> -> one flag per line
    local installer="$1" part scheme value ch
    while IFS= read -r part; do
        scheme="${part%%=*}"
        value="${part#*=}"
        [[ "${scheme}" == "${part}" ]] && value=""   # bare scheme, no '='
        if [[ "${installer}" == "pip" ]]; then
            case "${scheme}" in
                pypi)                 ;;
                index)  printf '%s\n' "--index-url"       "${value}" ;;
                extra)  printf '%s\n' "--extra-index-url" "${value}" ;;
                links)  printf '%s\n' "--find-links"      "${value}" ;;
                direct|git|local)     ;;   # handled by make_spec
                *) die "unknown pip source scheme '${scheme}' in manifest: $2" ;;
            esac
        else
            case "${scheme}" in
                channel)
                    local OLDIFS="${IFS}"; IFS=','
                    for ch in ${value}; do printf '%s\n' "-c" "${ch}"; done
                    IFS="${OLDIFS}"
                    ;;
                nodefaults) printf '%s\n' "--override-channels" ;;
                *) die "unknown conda source scheme '${scheme}' in manifest: $2" ;;
            esac
        fi
    done < <(split_source "$2")
}

# source_flags() runs inside a subshell (mapfile < <(...)), where die() can
# only kill the subshell -- the caller would carry on with empty flags and
# silently install from the wrong index. So every source string is validated
# here, in the main shell, before anything is installed.
validate_manifest() {
    local n=0 bad=0 line stage pkg ver arch installer source flags required note
    local part scheme
    while IFS=$'\t' read -r stage pkg ver arch installer source flags required note; do
        n=$((n + 1))
        case "${installer}" in
            pip|conda) ;;
            *) err "row '${pkg}': installer must be 'pip' or 'conda', got '${installer}'"; bad=1 ;;
        esac
        case "${required}" in
            yes|no) ;;
            *) err "row '${pkg}': required must be 'yes' or 'no', got '${required}'"; bad=1 ;;
        esac
        while IFS= read -r part; do
            scheme="${part%%=*}"
            if [[ "${installer}" == "pip" ]]; then
                case "${scheme}" in
                    pypi|index|extra|links|direct|git|local) ;;
                    *) err "row '${pkg}': unknown pip source scheme '${scheme}' in '${source}'"; bad=1 ;;
                esac
            else
                case "${scheme}" in
                    channel|nodefaults) ;;
                    *) err "row '${pkg}': unknown conda source scheme '${scheme}' in '${source}'"; bad=1 ;;
                esac
            fi
            # Schemes that take a value must actually have one.
            case "${scheme}" in
                index|extra|links|direct|git|local|channel)
                    [[ "${part}" == *=* && -n "${part#*=}" ]] \
                        || { err "row '${pkg}': source scheme '${scheme}' needs a value"; bad=1; } ;;
            esac
        done < <(split_source "${source}")
    done < <(rows)

    [[ ${n} -gt 0 ]] || die "manifest has no data rows: ${MANIFEST}"

    local pyrows
    pyrows="$(rows | awk -F'\t' '$1 == "00-env" && $2 == "python"')"
    [[ -n "${pyrows}" ]] || die "manifest has no 00-env row for python"
    [[ "$(printf '%s\n' "${pyrows}" | wc -l)" -eq 1 ]] \
        || die "manifest has more than one 00-env row for python"
    # That row becomes 'conda create', so an installer of 'pip' there would be
    # a statement the script cannot honour. Refuse it rather than ignore it.
    [[ "$(printf '%s' "${pyrows}" | cut -f5)" == "conda" ]] \
        || die "the 00-env row for python must have installer 'conda'; it becomes 'conda create'."

    [[ ${bad} -eq 0 ]] || die "manifest is invalid; fix the rows above."
    MANIFEST_ROWS=${n}
}

make_spec() {   # make_spec <installer> <package> <version> <source>
    local installer="$1" pkg="$2" ver="$3" part scheme value
    # An artifact-naming scheme wins over the version column.
    while IFS= read -r part; do
        scheme="${part%%=*}"; value="${part#*=}"
        case "${scheme}" in
            direct|local) printf '%s' "${value}"; return ;;
            git)          printf '%s @ git+%s' "${pkg}" "${value}"; return ;;
        esac
    done < <(split_source "$4")

    # conda's '==' demands an exact version string, so '==3.10' would NOT
    # match 3.10.20. conda wants a single '='; pip wants '=='.
    local eq="=="
    [[ "${installer}" == "conda" ]] && eq="="

    case "${ver}" in
        latest|-|match-torch) printf '%s' "${pkg}" ;;
        \>=*|\<=*|\>*|\<*|==*|!=*|~=*) printf '%s%s' "${pkg}" "${ver}" ;;
        *) printf '%s%s%s' "${pkg}" "${eq}" "${ver}" ;;
    esac
}

do_install() {
    local CONDA_BIN
    CONDA_BIN="$(find_conda)" || die "conda not found. Searched PATH, \$CONDA_EXE and the usual prefixes.
Install miniforge/miniconda first, or export CONDA_EXE=/path/to/conda."
    ok "Using conda: ${CONDA_BIN}"

    # --- open the run record -------------------------------------------------
    logf "# MAE-ST installer run record"
    logf "date:        $(date -Is)"
    logf "host:        $(hostname) ($(uname -m), ${HOST_OS})"
    logf "gpu:         ${HOST_GPU} (sm_${HOST_SM}, driver CUDA ${HOST_CUDA})"
    logf "manifest:    ${MANIFEST}"
    logf "manifest sha256: $(sha256sum "${MANIFEST}" 2>/dev/null | cut -d' ' -f1)"
    logf "env name:    ${ENV_NAME}"
    logf "conda:       ${CONDA_BIN} ($("${CONDA_BIN}" --version 2>&1))"
    logf "diverged:    ${DIVERGED}"
    logf ""

    # --- destructive confirmation -------------------------------------------
    if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        rule
        warn "An environment named '${ENV_NAME}' already exists."
        warn "Continuing will DELETE it and everything installed into it."
        rule
        if [[ ${NONINTERACTIVE} -eq 0 ]]; then
            local reply
            read -r -p "Type the environment name to confirm deletion: " reply
            [[ "${reply}" == "${ENV_NAME}" ]] || { say "Aborted; nothing changed."; return 1; }
        fi
        say "Removing existing environment '${ENV_NAME}'..."
        if [[ ${DRY_RUN} -eq 1 ]]; then
            say "  [dry-run] ${CONDA_BIN} env remove -n ${ENV_NAME} -y"
        else
            logf "\$ ${CONDA_BIN} env remove -n ${ENV_NAME} -y"
            "${CONDA_BIN}" env remove -n "${ENV_NAME}" -y
        fi
    fi

    # --- create the env from the 00-env/python row ---------------------------
    # The row is honoured in full: its version AND its source (channels). The
    # old version read only the version column and hardcoded the channel, so a
    # manifest saying 'conda-forge' silently got python from 'defaults'.
    # Read the row once, whole. Do NOT write this as 'rows | awk ... { exit }':
    # awk exiting early closes the pipe under the producer, which takes SIGPIPE,
    # and with 'set -o pipefail' that becomes exit 141 and kills the script.
    # It is a race, so it fails intermittently -- the worst way to find out.
    # validate_manifest has already guaranteed exactly one such row.
    local pyrow pyver pysource
    pyrow="$(rows | awk -F'\t' '$1 == "00-env" && $2 == "python"')"
    [[ -n "${pyrow}" ]] || die "manifest has no 00-env row for python"
    pyver="$(printf '%s' "${pyrow}" | cut -f3)"
    pysource="$(printf '%s' "${pyrow}" | cut -f6)"
    [[ -n "${pyver}" ]] || die "the 00-env row for python has no version"

    local -a pyflags=()
    mapfile -t pyflags < <(source_flags conda "${pysource}")

    local create_cmd=("${CONDA_BIN}" create -n "${ENV_NAME}")
    [[ ${#pyflags[@]} -gt 0 ]] && create_cmd+=("${pyflags[@]}")
    create_cmd+=("python=${pyver}" -y)

    ok "Creating environment '${ENV_NAME}' with python=${pyver} (source: ${pysource})..."
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '  [dry-run] '; printf '%q ' "${create_cmd[@]}"; printf '\n'
    else
        logf "\$ ${create_cmd[*]}"
        "${create_cmd[@]}"
        # shellcheck disable=SC1090
        eval "$("${CONDA_BIN}" shell.bash hook)"
        conda activate "${ENV_NAME}"
    fi

    # A silently failed 'conda activate' leaves you installing the entire
    # dependency set into base, which looks like success right up until it
    # breaks something else on the machine. Verify before touching pip.
    if [[ ${DRY_RUN} -eq 0 ]]; then
        local expected_py="${CONDA_PREFIX}/bin/python"
        [[ "$(command -v python)" == "${expected_py}" ]] \
            || die "conda activate did not take effect.
  expected python: ${expected_py}
  got:             $(command -v python)"
        ok "Environment active: $(python -V 2>&1) at ${CONDA_PREFIX}"
    fi

    # --- run the remaining stages -------------------------------------------
    local stage
    for stage in $(install_rows | awk -F'\t' '{ print $1 }' | sort -u); do
        rule
        say "${B}Stage ${stage}${N}"
        logf ""
        logf "# stage ${stage}"
        # Group rows sharing installer/source/flags/required into one command,
        # so torch and torchvision resolve together.
        local key
        while IFS= read -r key; do
            [[ -n "${key}" ]] || continue
            local installer source flags required specs=()
            IFS='|' read -r installer source flags required <<<"${key}"

            local pkg ver
            while IFS=$'\t' read -r pkg ver; do
                specs+=("$(make_spec "${installer}" "${pkg}" "${ver}" "${source}")")
            done < <(install_rows | awk -F'\t' -v s="${stage}" -v k="${key}" \
                        '$1 == s && ($5 "|" $6 "|" $7 "|" $8) == k { print $2 "\t" $3 }')

            [[ ${#specs[@]} -gt 0 ]] || continue

            local -a srcflags=()
            mapfile -t srcflags < <(source_flags "${installer}" "${source}")

            local cmd=()
            if [[ "${installer}" == "pip" ]]; then
                cmd=(python -m pip install)
            else
                cmd=("${CONDA_BIN}" install -n "${ENV_NAME}" -y)
            fi
            [[ ${#srcflags[@]} -gt 0 ]] && cmd+=("${srcflags[@]}")
            [[ "${flags}" != "-" ]] && cmd+=(${flags})
            cmd+=("${specs[@]}")

            if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '  [dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
                continue
            fi
            say "  \$ ${cmd[*]}"
            logf "\$ ${cmd[*]}"
            if "${cmd[@]}"; then
                ok "  ok: ${specs[*]}"
                logf "  -> ok"
            elif [[ "${required}" == "yes" ]]; then
                logf "  -> FAILED (required); aborting"
                die "required install failed: ${specs[*]}"
            else
                logf "  -> failed (optional, required=no); continuing"
                warn "  WARNING: optional install failed: ${specs[*]}"
                warn "  Continuing. See the manifest notes for why this may be expected."
            fi
        done < <(install_rows | awk -F'\t' -v s="${stage}" '$1 == s { print $5 "|" $6 "|" $7 "|" $8 }' | awk '!seen[$0]++')
    done

    rule
    if [[ ${DRY_RUN} -eq 1 ]]; then
        warn "Dry run complete. Nothing was installed."
        return 0
    fi

    write_lock "${CONDA_BIN}"

    # The lock is written BEFORE verify so that a failed verify still leaves a
    # record of what got installed -- that record is the only way to work out
    # why it failed.
    local rc=0
    verify || rc=$?
    logf ""
    logf "verify exit status: ${rc}"
    rule
    say "Run record:  ${LOG_FILE}"
    say "Lock file:   ${LOCK_FILE}"
    [[ ${rc} -eq 0 ]] || die "verification failed. See the run record above."
    return 0
}

# --- Resolution check --------------------------------------------------------
# Answers one question before any download starts: do the versions in this
# manifest exist on the indexes it names, and can pip satisfy them together?
#
# It resolves for the manifest's TARGET interpreter and platform, not this
# machine's, so a manifest can be checked from anywhere. Nothing is created and
# nothing is installed.
#
# What it CANNOT catch, and this is not a limitation that can be engineered
# away: a constraint nobody declared. torch 2.0.1 does not list numpy among its
# dependencies at all, yet it cannot run beside NumPy 2. Every resolver reports
# that pairing as fine. Only the functional test at the end of a real install,
# or prior knowledge written into a note, finds it.
# --platform must enumerate every tag a wheel might carry, and the list is
# open-ended: NVIDIA publishes manylinux_2_27, PyPI wheels use anything from
# 2_17 upward. A tag missing here shows up as a package that "cannot be found",
# which is indistinguishable from a real error. That is why this is used ONLY
# when checking a manifest written for a different architecture; on a matching
# host, pip's own tag set is authoritative and complete.
platform_flags() {   # platform_flags <arch> -> one --platform flag per line
    local a
    case "$1" in
        x86_64|amd64)  a=x86_64 ;;
        aarch64|arm64) a=aarch64 ;;
        *) printf -- '--platform\nany\n'; return 0 ;;
    esac
    local v
    printf -- '--platform\nlinux_%s\n' "${a}"
    printf -- '--platform\nmanylinux2014_%s\n' "${a}"
    for v in 2_17 2_24 2_27 2_28 2_31 2_34 2_35 2_36 2_39; do
        printf -- '--platform\nmanylinux_%s_%s\n' "${v}" "${a}"
    done
    printf -- '--platform\nany\n'
}

do_resolve() {
    local pybin
    pybin="$(command -v python3 || command -v python)" \
        || die "no python found to run pip with. --resolve-only needs any python 3 with pip."

    local tgt_py tgt_arch
    tgt_py="$(meta_get target python)"
    tgt_arch="$(meta_get target arch)"
    [[ -n "${tgt_py}" ]] || die "manifest #!target has no python=; --resolve-only needs it."
    # 'python=3.10.20' -> '3.10'; pip's --python-version takes major.minor.
    tgt_py="$(printf '%s' "${tgt_py}" | cut -d. -f1,2)"

    # Cross-architecture checking needs an explicit tag list, which can never be
    # complete; on a matching host, pip's own tags are authoritative. Prefer the
    # accurate mode whenever the architectures agree.
    local -a platflags=()
    local cross=0
    if [[ -n "${tgt_arch}" && "${tgt_arch}" != "${HOST_ARCH}" ]]; then
        cross=1
        mapfile -t platflags < <(platform_flags "${tgt_arch}")
    fi

    rule
    say "${B}Resolution check${N}"
    say "  resolving for      python ${tgt_py}, ${tgt_arch:-any arch}"
    say "  pip driving it     ${pybin}"
    say "  nothing is installed and no environment is created."
    if [[ ${cross} -eq 1 ]]; then
        warn "  APPROXIMATE: this manifest targets ${tgt_arch} and this host is"
        warn "  ${HOST_ARCH}, so wheels are matched against a hand-written list of"
        warn "  platform tags. A tag missing from that list looks exactly like a"
        warn "  missing package, so a failure here may be an artifact. Re-run on"
        warn "  the target machine to get a definitive answer."
    else
        say "  Architecture matches this host, so pip's own tag set is used."
    fi
    warn "  Binary-only: packages published solely as an sdist cannot be"
    warn "  resolved ahead of time and are reported as UNCHECKED, not failed."

    local failures=0 unchecked=0 stage
    for stage in $(install_rows | awk -F'\t' '{ print $1 }' | sort -u); do
        rule
        say "${B}Stage ${stage}${N}"
        local key
        while IFS= read -r key; do
            [[ -n "${key}" ]] || continue
            local installer source flags required specs=()
            IFS='|' read -r installer source flags required <<<"${key}"

            local pkg ver
            while IFS=$'\t' read -r pkg ver; do
                specs+=("$(make_spec "${installer}" "${pkg}" "${ver}" "${source}")")
            done < <(install_rows | awk -F'\t' -v s="${stage}" -v k="${key}" \
                        '$1 == s && ($5 "|" $6 "|" $7 "|" $8) == k { print $2 "\t" $3 }')
            [[ ${#specs[@]} -gt 0 ]] || continue

            if [[ "${installer}" != "pip" ]]; then
                say "  SKIP      ${specs[*]}"
                say "            conda rows are not resolved by this check."
                continue
            fi

            local -a srcflags=()
            mapfile -t srcflags < <(source_flags pip "${source}")

            local cmd=("${pybin}" -m pip install --dry-run --quiet --ignore-installed
                       --python-version "${tgt_py}" --only-binary=:all:)
            [[ ${#platflags[@]} -gt 0 ]] && cmd+=("${platflags[@]}")
            [[ ${#srcflags[@]} -gt 0 ]] && cmd+=("${srcflags[@]}")
            cmd+=("${specs[@]}")

            local out rc=0
            out="$("${cmd[@]}" 2>&1)" || rc=$?
            if [[ ${rc} -eq 0 ]]; then
                ok "  RESOLVES  ${specs[*]}"
            elif grep -q 'from versions: none' <<<"${out}"; then
                # No wheel at all: an sdist-only package. Not a manifest error.
                warn "  UNCHECKED ${specs[*]}"
                warn "            No wheel published, so it cannot be resolved without"
                warn "            building. Its version will be checked at install time."
                unchecked=$((unchecked + 1))
            elif [[ ${cross} -eq 1 ]]; then
                warn "  UNRESOLVED ${specs[*]}"
                printf '%s\n' "${out}" | sed 's/^/            /'
                warn "            Cross-architecture check: this may be a platform tag"
                warn "            missing from the list rather than a real problem."
                unchecked=$((unchecked + 1))
            elif [[ "${required}" == "yes" ]]; then
                err "  FAILS     ${specs[*]}"
                printf '%s\n' "${out}" | sed 's/^/            /' >&2
                failures=$((failures + 1))
            else
                warn "  FAILS (required=no)  ${specs[*]}"
                printf '%s\n' "${out}" | sed 's/^/            /'
            fi
        done < <(install_rows | awk -F'\t' -v s="${stage}" '$1 == s { print $5 "|" $6 "|" $7 "|" $8 }' | awk '!seen[$0]++')
    done

    rule
    if [[ ${failures} -eq 0 ]]; then
        ok "Every required row resolves. ${unchecked} row(s) unchecked (sdist only)."
        say "This says the versions exist and are mutually satisfiable."
        say "It does NOT say the environment will work: undeclared constraints,"
        say "such as a torch/NumPy ABI mismatch, are invisible here. The install's"
        say "own verification step is what settles that."
        return 0
    fi
    err "${failures} required row(s) could not be resolved. Fix the manifest before installing."
    return 1
}

# --- Reproducibility artifacts ----------------------------------------------
write_lock() {   # write_lock <conda_bin>
    local conda_bin="$1"
    say "${B}Recording resolved environment${N}"
    {
        printf '# pip freeze of conda env "%s"\n' "${ENV_NAME}"
        printf '# host:     %s (%s)\n' "$(hostname)" "$(uname -m)"
        printf '# date:     %s\n' "$(date -Is)"
        printf '# manifest: %s (sha256 %s)\n' "$(basename "${MANIFEST}")" \
               "$(sha256sum "${MANIFEST}" 2>/dev/null | cut -d' ' -f1)"
        printf '# python:   %s\n' "$(python -V 2>&1)"
        printf '#\n'
        printf '# This is the RESULT of a resolution, not a request. It records\n'
        printf '# transitive dependencies the manifest never names. To pin what\n'
        printf '# works, read versions from here into the manifest by hand -- do\n'
        printf '# not feed this file back to the installer.\n'
        printf '#\n'
        # 'pip list --format=freeze', NOT 'pip freeze'. Two reasons, both of
        # which cost a lock its whole purpose:
        #   - pip freeze renders conda-installed packages as
        #     'packaging @ file:///home/conda/feedstock_root/...', a path on
        #     the conda-forge BUILD machine. Unusable as a record.
        #   - pip freeze omits pip/setuptools/wheel unless asked. The
        #     'setuptools<70' pin is deliberate, so a record that hides it is
        #     worse than no record.
        python -m pip list --format=freeze
    } >"${LOCK_FILE}"
    ok "  wrote ${LOCK_FILE} ($(grep -cv '^#' "${LOCK_FILE}") packages)"

    logf ""
    logf "# conda list"
    "${conda_bin}" list -n "${ENV_NAME}" 2>/dev/null | sed 's/^/  /' >>"${LOG_FILE}" || true
}

# --- Post-install verification ----------------------------------------------
verify() {
    say "${B}Verifying installation${N}"
    rule
    MANIFEST="${MANIFEST}" python - <<'PY'
import importlib, os, sys

# package name in the manifest -> module name to import
MODMAP = {
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "pytorchvideo": "pytorchvideo.transforms",
    "torch": "torch", "torchvision": "torchvision",
}
SKIP = {"python", "pip", "setuptools", "wheel"}

rows = []
with open(os.environ["MANIFEST"]) as fh:
    for line in fh:
        if line.startswith("#") or "\t" not in line:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[0] == "stage":
            continue
        rows.append((f[1], f[7]))

failed_required = []
for pkg, required in rows:
    if pkg in SKIP:
        continue
    mod = MODMAP.get(pkg, pkg.replace("-", "_"))
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ok       {pkg:<26} {ver}")
    except Exception as exc:
        tag = "MISSING " if required == "yes" else "optional"
        print(f"  {tag} {pkg:<26} {type(exc).__name__}: {exc}")
        if required == "yes":
            failed_required.append(pkg)

print()
import torch
print(f"  torch            {torch.__version__}")
print(f"  built for CUDA   {torch.version.cuda}")
print(f"  cuda available   {torch.cuda.is_available()}")
arches = torch.cuda.get_arch_list()
print(f"  arch list        {arches}")

def parse_arch(tag):
    """'sm_121' -> (12, 1). The LAST digit is the minor, the rest is the major,
    which is why sm_100 is (10, 0) and not (1, 0, 0)."""
    digits = tag.split("_", 1)[1]
    return int(digits[:-1]), int(digits[-1])

if torch.cuda.is_available():
    print(f"  device           {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"  capability       {sm}")

    # A cubin built for sm_XY runs on any device of the same major X whose
    # minor is >= Y (CUDA minor-version binary compatibility). So an exact
    # string match is NOT the criterion: a torch whose arch list stops at
    # sm_120 runs natively on this sm_121 GB10, which is exactly the case on
    # the DGX Spark. The old check reported that healthy install as broken.
    cubins = [a for a in arches if a.startswith("sm_")]
    ptx    = [a for a in arches if a.startswith("compute_")]

    covering = [a for a in cubins
                if parse_arch(a)[0] == cap[0] and parse_arch(a)[1] <= cap[1]]
    jitting  = [a for a in ptx
                if parse_arch(a)[0] < cap[0]
                or (parse_arch(a)[0] == cap[0] and parse_arch(a)[1] <= cap[1])]

    if sm in cubins:
        print(f"  kernel coverage  native ({sm} built in)")
    elif covering:
        best = max(covering, key=parse_arch)
        print(f"  kernel coverage  binary-compatible via {best}")
        print(f"                   ({best} cubins run on {sm}: same major, lower minor)")
    elif jitting:
        print(f"  kernel coverage  PTX JIT only ({', '.join(jitting)})")
        print( "  WARNING: no prebuilt cubin for this chip. It will work, but the")
        print( "  first launch of every kernel pays a JIT compile.")
    else:
        # The failure mode this catches: torch imports fine, reports
        # cuda.is_available() == True, and still has no kernels for the chip.
        # Only the arch list and the functional test below reveal it.
        print(f"\n  WARNING: nothing in torch's arch list covers {sm}. Kernels")
        print( "  will fail. Pick a different torch index URL in the manifest.")

    # Whatever the arch list says, this is the verdict.
    x = torch.randn(64, 64, device="cuda")
    print(f"  matmul on gpu    {(x @ x).sum().item():.4f}")
    import torch.nn as nn
    # Conv3d is the layer mae_st's PatchEmbed is built on; exercise it too.
    conv = nn.Conv3d(3, 8, (2, 16, 16), stride=(2, 16, 16)).cuda()
    out = conv(torch.randn(1, 3, 16, 224, 224, device="cuda"))
    print(f"  conv3d on gpu    {tuple(out.shape)}")
else:
    print("\n  WARNING: no CUDA device visible to torch.")

if failed_required:
    print(f"\n  FAILED (required): {', '.join(failed_required)}")
    sys.exit(1)
PY
    rule
    ok "Done. Activate with:  conda activate ${ENV_NAME}"
}

# --- Menu --------------------------------------------------------------------
main_menu() {
    while true; do
        say ""
        say "${B}Menu${N}"
        say "  1) Change target environment name  (current: ${ENV_NAME})"
        say "  2) Show the dependency manifest"
        say "  3) Check that the manifest resolves  (downloads nothing)"
        say "  4) Install"
        say "  q) Quit"
        say ""
        local choice
        read -r -p "Choice: " choice
        case "${choice}" in
            1)
                local newname
                read -r -p "New environment name [${ENV_NAME}]: " newname
                if [[ -n "${newname}" ]]; then
                    if [[ "${newname}" =~ ^[A-Za-z0-9._-]+$ ]]; then
                        ENV_NAME="${newname}"
                        ok "Environment name set to '${ENV_NAME}'."
                    else
                        err "Invalid name. Use letters, digits, dot, dash, underscore."
                    fi
                fi
                ;;
            2) show_manifest ;;
            3) do_resolve || true ;;
            4) do_install && return 0 ;;
            q|Q) say "Nothing changed."; return 0 ;;
            *) err "Pick 1, 2, 3, 4 or q." ;;
        esac
    done
}

validate_manifest
show_preamble
[[ ${DRY_RUN} -eq 1 ]] && warn "  DRY RUN: commands will be printed, nothing executed."
if [[ ${RESOLVE_ONLY} -eq 1 ]]; then
    do_resolve
elif [[ ${NONINTERACTIVE} -eq 1 ]]; then
    do_install
else
    main_menu
fi
