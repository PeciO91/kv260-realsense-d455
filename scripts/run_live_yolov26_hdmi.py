import argparse
import os
import subprocess
import sys

DEFAULT_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "inference",
    "live_yolov26_hdmi",
)

LAUNCHER_OPTIONS = ("--binary", "--display")


def split_launcher_args(argv):
    launcher_args = []
    forwarded = []
    i = 0
    while i < len(argv):
        if argv[i] == "--":
            forwarded.extend(argv[i + 1:])
            break
        if argv[i] in (
            "--check-display", "--wake-display",
            "--help", "-h",
        ):
            launcher_args.append(argv[i])
            i += 1
            continue
        if argv[i] in LAUNCHER_OPTIONS:
            if i + 1 >= len(argv):
                raise ValueError(
                    "missing value for %s" % argv[i]
                )
            launcher_args.extend(argv[i:i + 2])
            i += 2
            continue
        if any(argv[i].startswith(o + "=") for o in LAUNCHER_OPTIONS):
            launcher_args.append(argv[i])
            i += 1
            continue
        forwarded.extend(argv[i:])
        break
    return launcher_args, forwarded


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="run_live_yolov26_hdmi.py",
        description="Launch the native live_yolov26_hdmi"
        " fullscreen display on the local X server",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=DEFAULT_BINARY,
    )
    parser.add_argument(
        "--display",
        type=str,
        default=":0",
        help="X display for check/wake actions and the"
        " child environment (default :0)",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check-display",
        action="store_true",
        help="query XRandR and DPMS, then exit without"
        " starting vision",
    )
    actions.add_argument(
        "--wake-display",
        action="store_true",
        help="force DPMS on and reset the screen saver, then"
        " exit; desktop power-saving policy is unchanged",
    )
    return parser


def run_display_tool(env, *argv):
    return subprocess.run(
        list(argv),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def child_env(display):
    env = dict(os.environ)
    env["DISPLAY"] = display
    return env


def check_display(display):
    env = child_env(display)
    rc = 0
    for argv in (("xrandr", "--query"), ("xset", "q")):
        try:
            result = run_display_tool(env, *argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(
                "error: cannot run %s: %s" % (argv[0], exc),
                file=sys.stderr,
            )
            return 1
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            rc = 1
    return rc


def wake_display(display):
    env = child_env(display)
    commands = (
        ("xset", "dpms", "force", "on"),
        ("xset", "s", "reset"),
        ("xset", "q"),
    )
    for argv in commands:
        try:
            result = run_display_tool(env, *argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(
                "error: cannot run %s: %s" % (argv[0], exc),
                file=sys.stderr,
            )
            return 1
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return 1
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        launcher_args, forwarded = split_launcher_args(list(argv))
        ns = build_arg_parser().parse_args(launcher_args)
    except (ValueError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            return exc.code if isinstance(exc.code, int) else 2
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if ns.check_display or ns.wake_display:
        if forwarded:
            print(
                "error: display actions take no vision"
                " arguments: %s" % " ".join(forwarded),
                file=sys.stderr,
            )
            return 2

    if ns.check_display:
        return check_display(ns.display)

    if ns.wake_display:
        return wake_display(ns.display)

    if not os.path.isfile(ns.binary):
        print(
            "error: binary not found: %s" % ns.binary,
            file=sys.stderr,
        )
        return 2

    argv_exec = [ns.binary, "--display", ns.display] + forwarded
    env = child_env(ns.display)
    try:
        os.execvpe(ns.binary, argv_exec, env)
    except OSError as exc:
        print(
            "error: cannot exec %s: %s" % (ns.binary, exc),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
