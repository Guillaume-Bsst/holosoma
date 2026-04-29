# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi
source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate hscinference
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/hscinference/lib/python3.11/site-packages/lib

_PYTHON=${CONDA_ROOT}/envs/hscinference/bin/python

# Ensure holosoma_inference is installed (editable mode)
if ! ${_PYTHON} -c "import holosoma_inference" 2>/dev/null; then
    echo "holosoma_inference not found — installing in editable mode..."
    ${_PYTHON} -m pip install -e ${SCRIPT_DIR}/../src/holosoma_inference
fi

# Ensure tyro is installed
if ! ${_PYTHON} -c "import tyro" 2>/dev/null; then
    echo "tyro not found — installing..."
    ${_PYTHON} -m pip install tyro
fi

# Check UFW status if ufw command exists (no sudo required)
if command -v ufw >/dev/null 2>&1; then
    if ufw status 2>/dev/null | grep -q "Status: inactive"; then
        echo "✓ UFW disabled"
    else
        echo "Warning: UFW may be enabled (run 'sudo ufw status' to check)."
    fi
fi
