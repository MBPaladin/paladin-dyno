import os

# Repo root: override with DYNO_ROOT if set, otherwise derive from this file's
# location (deployment/ sits directly under the repo root). On the dyno host
# this resolves to the same /home/paladin/Documents/Github/paladin-dyno as the
# old hardcoded value; in sandboxes/clones it just works.
working_directory = os.environ.get(
    'DYNO_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dyno_directory = f"{working_directory}/dyno"
dyno_src_directory = f"{dyno_directory}/src"
dyno_test_directory = f"{dyno_directory}/tests"
dyno_config_directory = f"{dyno_directory}/config"
dyno_logs_directory = f"{dyno_directory}/logs"
