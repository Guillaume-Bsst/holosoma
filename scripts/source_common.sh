WORKSPACE_DIR=${WORKSPACE_DIR:-$HOME/.holosoma_custom_deps}
CONDA_ROOT=$WORKSPACE_DIR/miniconda3

# Fully clean any conda state from the current shell to prevent PATH pollution
# from other conda installations (e.g. miniforge3, miniconda3, system conda).
while [ "${CONDA_SHLVL:-0}" -gt 0 ] 2>/dev/null; do
    conda deactivate 2>/dev/null || break
done
# Strip any residual conda/mamba paths so only the holosoma miniconda is used
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/miniforge3/' | grep -v '/miniconda3/' | tr '\n' ':' | sed 's/:$//')
unset CONDA_EXE _CONDA_EXE _CONDA_ROOT CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER

if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi
