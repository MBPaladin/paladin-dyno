source .venv/bin/activate
# DYNO_CODE_CLI: resolve the VS Code CLI before sudo strips the user PATH,
# so the GUI's safety "edit" links can open the config in the editor.
sudo env DYNO_CODE_CLI="$(command -v code || true)" .venv/bin/python src/gui.py --actuator
