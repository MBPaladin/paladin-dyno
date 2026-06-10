source .venv/bin/activate
sudo chown -R "$(whoami)" logs
.venv/bin/python src/post_processor.py "$@"
