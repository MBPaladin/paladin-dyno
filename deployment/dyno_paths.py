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


def resolved_config_path():
    """Where Master writes the bring-up's resolved config and Logger reads it.

    A sim session (DYNO_SIM set) gets its own file. Both share this one
    directory, and the rig's Logger stamps whatever is in the file onto the
    next log it opens, so a sim brought up beside a live rig session would
    otherwise put the sim's configuration into the rig's next log.
    """
    name = 'resolved_config.sim.json' if os.environ.get('DYNO_SIM') else 'resolved_config.json'
    return f"{dyno_logs_directory}/{name}"
