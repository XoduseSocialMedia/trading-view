"""
FinxView — Custom Script sandbox runner.

This file is executed as a SEPARATE OS PROCESS (via subprocess.Popen in
main.py), never imported into the main FastAPI process. That process
boundary is what actually makes the sandbox real:

  - A script that hangs (infinite loop) or is killed for exceeding its
    wall-clock timeout only kills THIS process — the main server and every
    other user's request keep running untouched.
  - A script that calls sys.exit(), crashes the interpreter, or exhausts
    memory only takes this process down with it.
  - Restricted builtins/imports (below) stop the easy, obvious escapes
    (import os/subprocess/socket, open(), eval of arbitrary strings, etc.)
    as defense-in-depth on top of the process boundary.

Honesty about limits (see item 20 of the brief: never claim "impossible to
hack"): this is a strong, real improvement over exec()-in-a-thread, not a
hardened container. On Linux, RLIMIT_AS/RLIMIT_CPU below add real memory/
CPU ceilings. On Windows (which is how this app is actually distributed —
see start.bat) the `resource` module doesn't exist at all, so on Windows
this relies on the wall-clock subprocess timeout + restricted builtins
only, not a hard memory ceiling. A determined attacker with the ability to
submit scripts to a Windows deployment could still allocate a lot of
memory before the timeout fires. If this is ever exposed to untrusted
users on Windows, run it behind a Linux container (Docker/WSL2) so the
RLIMIT path below actually applies.

Contract with main.py:
  argv[1] = path to a .txt file containing the exact user script source
  argv[2] = path to a .csv file with columns time,open,high,low,close,volume
  argv[3] = path to write a JSON result file to (created by this process)

  stdout  = ONLY the user script's own print() output (nothing else is
            ever written there, so main.py can show it to the user as-is)
  stderr  = diagnostic info if this runner itself fails before producing
            a result file (main.py only surfaces this when result.json is
            missing entirely)
  exit code 0  = result.json was written successfully (even if the
                 script's *logic* raised — that's reported inside the
                 JSON as {"error": ...}, not via exit code, so main.py can
                 tell "runner infrastructure failed" apart from "user
                 script raised")
  exit code !=0 = the runner crashed before it could write result.json at
                 all (e.g. resource limit killed it, syntax error while
                 parsing argv) — main.py falls back to stderr text
"""
import sys
import os
import json
import math
import time
import re


class _NumpyJSONEncoder(json.JSONEncoder):
    """pandas/numpy operations naturally produce numpy scalar types
    (df['x'].iloc[i] is numpy.int64, not a plain Python int) — completely
    normal, expected usage that a script author has no reason to think
    twice about — but json.dump() has no built-in support for them.
    Converting them here means an ordinary, correct pandas script never
    fails for a reason unrelated to its actual logic."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if pd.isna(obj):
                return None
        except (ImportError, TypeError, ValueError):
            pass
        return super().default(obj)


def _write_result(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_NumpyJSONEncoder)


def _apply_resource_limits(mem_mb: int, cpu_seconds: int):
    """Best-effort, POSIX-only. See module docstring for the Windows gap."""
    if os.name != "posix":
        return
    try:
        import resource
        mem_bytes = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass  # some platforms (notably macOS) don't support RLIMIT_AS — degrade silently
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass
        try:
            # No reason a chart-indicator script ever needs to spawn processes
            # or open more than a handful of files at once.
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except Exception:
            pass
    except ImportError:
        pass


# Modules a legitimate indicator script has real reasons to import. Anything
# not in this list is blocked — in particular os/sys/subprocess/socket/
# shutil/pathlib/ctypes/importlib/multiprocessing/threading/pickle and any
# networking library (urllib/http/requests), which have no place in a
# chart-math script and are the actual escape routes worth blocking.
_ALLOWED_TOP_LEVEL_IMPORTS = {
    "pandas", "numpy", "scipy", "sklearn", "statsmodels", "ta", "seaborn",
    "matplotlib", "math", "statistics", "itertools", "collections",
    "functools", "datetime", "json", "re", "decimal", "random", "typing",
    "warnings", "copy", "heapq", "bisect", "array", "time", "contextlib",
    "string", "fractions", "uuid", "enum", "dataclasses",
}


def _make_yfinance_stub(chart_frame):
    """Scripts written for the command line very often start by downloading
    their own candles (`yf.download(SYMBOL, ...)`) — but the whole point of
    running inside the app is that the candles are already here: the chart
    you're looking at. Real yfinance can't be allowed (it's a network
    client, and the sandbox deliberately has none), so scripts get this
    stand-in: download()/Ticker().history() hand back THE CHART'S OWN
    CANDLES in yfinance's usual shape (a DatetimeIndex named "Date" and
    Open/High/Low/Close/Adj Close/Volume columns). No network is ever
    touched. Consequence worth knowing: the symbol / interval / start / end
    arguments are accepted but IGNORED — a script always gets whatever
    symbol and timeframe is currently on the chart, which is exactly what
    an indicator drawing on that chart needs."""
    import types
    import pandas as pd

    def _col(*names):
        for n in names:
            if n in chart_frame.columns:
                return chart_frame[n].to_numpy()
        return None

    def _build():
        t = chart_frame["time"] if "time" in chart_frame.columns else chart_frame["Date"]
        if pd.api.types.is_numeric_dtype(t):
            idx = pd.to_datetime(t.to_numpy(), unit="s")      # intraday candles: unix seconds
        else:
            idx = pd.to_datetime(t.to_numpy())                # daily/weekly/monthly: date strings
        close = _col("close", "Close")
        vol = _col("volume", "Volume")
        return pd.DataFrame({
            "Open": _col("open", "Open"), "High": _col("high", "High"),
            "Low": _col("low", "Low"), "Close": close, "Adj Close": close,
            "Volume": vol if vol is not None else 0,
        }, index=pd.DatetimeIndex(idx, name="Date"))

    class Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, *a, **k):
            return _build()

    mod = types.ModuleType("yfinance")
    mod.download = lambda *a, **k: _build()
    mod.Ticker = Ticker
    return mod


def _make_argparse_stub():
    """Standalone scripts very commonly `import argparse` at the top for a
    command-line block under `if __name__ == "__main__":` — a block that
    never runs here (the script's __name__ is "finxview_script"). The real
    argparse can't simply be allow-listed: it exposes its own `_os`/`_sys`
    module handles (argparse._os.system(...) etc.), which would hand a
    script the exact escape routes this sandbox exists to block. So scripts
    get this inert stand-in instead: it accepts the same add_argument()
    calls and parse_args() returns each option's DEFAULT value (never
    reading the real command line), which is what a script run with no
    arguments would get anyway. No module handles, no I/O."""
    import types

    mod = types.ModuleType("argparse")

    class Namespace:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __repr__(self):
            return "Namespace(" + ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()) + ")"

    class ArgumentParser:
        def __init__(self, *a, **k):
            self._defaults = []

        def add_argument(self, *names, **kw):
            action = kw.get("action")
            dest = kw.get("dest")
            if dest is None:
                opts = [n for n in names if str(n).startswith("-")]
                if opts:
                    longs = [n for n in opts if n.startswith("--")]
                    dest = (longs[0] if longs else opts[0]).lstrip("-").replace("-", "_")
                else:
                    dest = names[0]
            if action == "store_true":
                default = kw.get("default", False)
            elif action == "store_false":
                default = kw.get("default", True)
            else:
                default = kw.get("default")
                conv = kw.get("type")
                if isinstance(default, str) and callable(conv):
                    try:
                        default = conv(default)
                    except Exception:
                        pass
            self._defaults.append((dest, default))

        def set_defaults(self, **kw):
            for dest, value in kw.items():
                self._defaults.append((dest, value))

        def parse_args(self, args=None, namespace=None):
            ns = namespace if namespace is not None else Namespace()
            for dest, default in self._defaults:
                setattr(ns, dest, default)
            return ns

        def parse_known_args(self, args=None, namespace=None):
            return self.parse_args(args, namespace), []

        # Groups just forward to the same parser so .add_argument() keeps working.
        def add_argument_group(self, *a, **k):
            return self

        def add_mutually_exclusive_group(self, *a, **k):
            return self

        def print_help(self, *a, **k):
            pass

        def print_usage(self, *a, **k):
            pass

        def error(self, message):
            raise ValueError(message)

    mod.ArgumentParser = ArgumentParser
    mod.Namespace = Namespace
    return mod


_ARGPARSE_STUB = []  # built lazily, once


def _make_guarded_import(real_import, chart_frame=None):
    yf_stub = []  # built lazily, once per script run

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".", 1)[0]
        if top == "yfinance" and chart_frame is not None:
            if not yf_stub:
                yf_stub.append(_make_yfinance_stub(chart_frame))
            return yf_stub[0]
        if top == "argparse":
            if not _ARGPARSE_STUB:
                _ARGPARSE_STUB.append(_make_argparse_stub())
            return _ARGPARSE_STUB[0]
        if top not in _ALLOWED_TOP_LEVEL_IMPORTS:
            raise ImportError(
                f"'{name}' is not available in Custom Script sandbox. "
                f"Allowed: {', '.join(sorted(_ALLOWED_TOP_LEVEL_IMPORTS))}"
            )
        return real_import(name, globals, locals, fromlist, level)
    return guarded_import


def _build_restricted_globals(pd, np, frame, captured_print, as_main=False):
    """A minimal, curated builtins set. Notably absent on purpose: open,
    eval, exec, compile, __import__ (replaced with the guarded version),
    globals/locals/vars, input, exit/quit, breakpoint, memoryview."""
    safe_names = [
        "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "callable",
        "chr", "classmethod", "complex", "delattr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "getattr",
        "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
        "pow", "property", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
        "tuple", "type", "zip", "True", "False", "None", "NotImplemented",
        "Ellipsis", "__build_class__",
        # Exception/warning types a script might reasonably want to catch
        # or raise — this is language completeness, not a security
        # boundary; none of these grant any capability the sandbox
        # doesn't already allow.
        "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "ZeroDivisionError", "StopIteration",
        "StopAsyncIteration", "RuntimeError", "ArithmeticError",
        "OverflowError", "NameError", "AssertionError", "NotImplementedError",
        "FloatingPointError", "LookupError", "OSError", "IOError",
        "ImportError", "ModuleNotFoundError", "RecursionError", "MemoryError",
        "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError",
        "UnboundLocalError", "Warning", "UserWarning", "DeprecationWarning",
        "FutureWarning", "RuntimeWarning",
    ]
    import builtins as _b
    restricted_builtins = {n: getattr(_b, n) for n in safe_names if hasattr(_b, n)}
    restricted_builtins["__import__"] = _make_guarded_import(_b.__import__, frame)
    restricted_builtins["print"] = captured_print

    ns = {
        "__builtins__": restricted_builtins,
        "pd": pd, "np": np, "df": frame,
        # "__main__" only for the fallback run of a script's own
        # `if __name__ == "__main__":` block — see main().
        "__name__": "__main__" if as_main else "finxview_script",
        "print": captured_print,
    }

    # Pine-Script-style bare series names and bar_index — scripts adapted
    # from Pine Script very naturally assume these are always available
    # with no import and no df[...] lookup, exactly like Pine Script
    # itself provides them; "not defined" for one of these is a common,
    # avoidable failure otherwise. Always exposed under these lowercase
    # names regardless of which column-naming convention `frame` itself
    # uses (open/High/etc. — the two attempts this runner tries), since
    # Pine's own built-ins are always lowercase. A script assigning these
    # names itself (e.g. `close = df["close"].rolling(5).mean()`) simply
    # rebinds them in the normal way — nothing here prevents that.
    def _bare_col(*names):
        for name in names:
            if name in frame.columns:
                return frame[name].to_numpy()
        return None
    ns["open"] = _bare_col("open", "Open")
    ns["high"] = _bare_col("high", "High")
    ns["low"] = _bare_col("low", "Low")
    ns["close"] = _bare_col("close", "Close")
    ns["volume"] = _bare_col("volume", "Volume")
    ns["bar_index"] = np.arange(len(frame))

    return ns


def main():
    if len(sys.argv) != 4:
        print("usage: sandbox_runner.py <code_file> <data_csv> <result_json>", file=sys.stderr)
        sys.exit(2)

    code_path, data_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Read from env rather than hardcoding here a second time — main.py sets
    # these based on the same SCRIPT_TIMEOUT_SECONDS it uses for the
    # subprocess wall-clock timeout, so the CPU limit always fires a little
    # BEFORE the wall-clock kill for a pure CPU-bound loop (a faster, more
    # specific death: SIGXCPU with a clear reason vs. a generic SIGKILL).
    mem_mb = int(os.environ.get("FINXVIEW_SCRIPT_MEM_MB", "512"))
    cpu_seconds = int(os.environ.get("FINXVIEW_SCRIPT_CPU_SECONDS", "25"))
    _apply_resource_limits(mem_mb=mem_mb, cpu_seconds=cpu_seconds)

    # Imported AFTER the resource limits are set, deliberately — pandas/
    # numpy's own import-time allocations then count against the same cap
    # a runaway script would be held to, which is the honest behavior.
    import pandas as pd
    import numpy as np

    # Force a headless matplotlib backend BEFORE the user's script can
    # import pyplot itself. Without this, matplotlib auto-detects the
    # backend — on this Linux sandbox (no display) that's already headless
    # 'agg' by luck, but on a real desktop (Windows, where this app
    # actually ships and runs via start.bat) a display genuinely exists,
    # so matplotlib picks an INTERACTIVE backend instead. A script calling
    # plt.show() then opens a real desktop window and BLOCKS this entire
    # subprocess until a human manually closes it — which looks exactly
    # like "the script never finishes" from the app's side, and the
    # script's actual output (its `events`/`result` variable) never gets
    # returned at all, since exec() never returns while show() is blocking.
    # Setting the backend here, before any user code runs, means
    # plt.show() becomes a harmless no-op everywhere — every other
    # matplotlib call (building the figure, plt.plot, Rectangle, etc.)
    # still works completely normally under this backend; only the
    # interactive-popup behavior is disabled, which is exactly what a
    # server-side sandbox should never do in the first place.
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except ImportError:
        pass  # matplotlib not installed — nothing to neutralize

    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"could not read script file: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        raw = pd.read_csv(data_path)
    except Exception as e:
        print(f"could not read chart data: {e}", file=sys.stderr)
        sys.exit(2)

    lower_df = raw[["time", "open", "high", "low", "close", "volume"]].copy()
    cap_df = lower_df.rename(columns={
        "time": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })

    printed_parts = []

    def _captured_print(*args, sep=" ", end="\n", **kwargs):
        printed_parts.append(sep.join(str(a) for a in args) + end)

    def _exec_attempt(frame, as_main=False):
        # Every real DataFrame method (pd.read_csv redirection included)
        # stays exactly as before — only the exec() *namespace* the user's
        # top-level code runs in is restricted, so ordinary pandas/numpy
        # calls the script makes are completely unaffected.
        original_read_csv = pd.read_csv
        original_read_excel = pd.read_excel
        pd.read_csv = lambda *a, **k: frame.copy()
        pd.read_excel = lambda *a, **k: frame.copy()
        try:
            ns = _build_restricted_globals(pd, np, frame, _captured_print, as_main)
            exec(code, ns)
            return ns
        finally:
            pd.read_csv = original_read_csv
            pd.read_excel = original_read_excel

    def _looks_like_marks_list(v):
        if not isinstance(v, list) or len(v) == 0:
            return False
        for item in v:
            if not isinstance(item, dict):
                return False
            if not any(k in item for k in ("price", "level")):
                return False
            if not any(k in item for k in ("start", "from_bar", "bar")):
                return False
        return True

    import inspect
    import types

    def _find_auto_callable_function(namespace):
        """A function is a candidate if it's user-defined (came from this
        exec, not something imported), isn't run_smc (already handled
        separately above), and takes exactly one required positional
        argument — the natural shape of "a function that transforms the
        candle dataframe". Only auto-called when there's exactly one such
        function, so this never has to guess between several."""
        candidates = []  # (function, name of its one required parameter)
        for name, val in namespace.items():
            if name == "run_smc" or not isinstance(val, types.FunctionType):
                continue
            try:
                sig = inspect.signature(val)
            except (ValueError, TypeError):
                continue
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(required) == 1:
                candidates.append((val, required[0].name.lower()))
        if len(candidates) == 1:
            return candidates[0][0]
        # More than one function takes a single argument — e.g. analyze(df)
        # alongside load(path), a very ordinary pairing in a standalone
        # script. The parameter NAME is what tells them apart: only the one
        # that asks for the candle data (df/data/candles/...) is something
        # this app can actually call. If that still leaves zero or several,
        # don't guess.
        df_like = {"df", "data", "candles", "ohlc", "ohlcv", "bars", "prices",
                   "frame", "dataframe", "data_frame", "dff"}
        narrowed = [fn for fn, pname in candidates if pname in df_like]
        return narrowed[0] if len(narrowed) == 1 else None

    def _interpret_auto_result(value):
        """Best-effort interpretation of whatever the auto-called function
        returned, covering every shape already documented for scripts that
        wire things up explicitly: an (events, hashes) tuple, a dict with
        "lines", a bare list of event dicts, or — the one genuinely new
        case, since nothing already handles it — a DataFrame with one or
        more "..._level" columns (a price at the rows a pattern was
        detected, NaN everywhere else), which gets turned into a marker at
        each non-null row."""
        if value is None:
            return None
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list):
            return {"kind": "smc_events", "value": value[0]}
        if isinstance(value, dict) and isinstance(value.get("lines"), list):
            return {"kind": "result", "value": value}
        if isinstance(value, list) and len(value) > 0 and all(isinstance(x, dict) for x in value):
            # Dicts with a price/level + start bar are marker lists (same test
            # applied to a script's own `events` variable above) — routed
            # there so their own labels/kinds survive. Anything else keeps
            # the previous smc_events handling.
            if _looks_like_marks_list(value):
                return {"kind": "marks", "value": value}
            return {"kind": "smc_events", "value": value}
        if isinstance(value, pd.DataFrame):
            level_cols = [c for c in value.columns if isinstance(c, str) and c.lower().endswith("_level")]
            if not level_cols:
                return None
            vdf = value.reset_index(drop=True)
            lines = []
            for col in level_cols:
                label = col[: -len("_level")].upper()
                for i, lvl in enumerate(vdf[col]):
                    try:
                        if pd.isna(lvl):
                            continue
                    except (TypeError, ValueError):
                        continue
                    lines.append({"from_bar": i, "to_bar": i, "price": float(lvl), "label": label})
            return {"kind": "result", "value": {"lines": lines}} if lines else None
        return None

    def _run_both_conventions(as_main):
        """Lowercase column names first, then (only if that raises) the
        capitalized yfinance-style ones — see the longer note on why these
        are never both present at once. Same behavior as before; pulled
        into a function so the fallback __main__ run can reuse it."""
        try:
            return _exec_attempt(lower_df, as_main)
        except Exception as first_err:
            try:
                return _exec_attempt(cap_df, as_main)
            except Exception as second_err:
                if str(first_err) == str(second_err):
                    # Same failure under both column-naming conventions —
                    # one message is the clearest thing to show.
                    raise first_err
                # Different failures: attempt 2 got further before hitting
                # its own, unrelated problem, so it's almost always the
                # more relevant one to lead with — but the first is kept
                # too rather than silently discarded, in case it's actually
                # the useful one (e.g. a genuine typo'd column name that
                # happens to surface differently under each convention).
                raise Exception(
                    f"{second_err}\n\n(Also failed differently under the other "
                    f"column-naming convention: {first_err})"
                )

    def _detect(namespace, printed):
        """Looks through what a finished run left behind and returns the
        dict to hand back to main.py, an {"error": ...} dict, or None when
        there's genuinely nothing to draw. (Same checks, same order as
        before — only moved into a function so they can be applied to a
        second run.)"""
        if isinstance(namespace.get("result"), dict):
            return {"kind": "result", "value": namespace["result"], "printed": printed}

        if callable(namespace.get("run_smc")):
            try:
                events, hashes = namespace["run_smc"](lower_df)
            except TypeError:
                events, hashes = namespace["run_smc"](lower_df, 20, 4)
            except Exception as e:
                return {"error": f"run_smc raised: {str(e)[:400]}", "printed": printed}
            return {"kind": "smc_events", "value": events, "printed": printed}

        ev = namespace.get("events")
        if isinstance(ev, list) and len(ev) > 0:
            if _looks_like_marks_list(ev):
                return {"kind": "marks", "value": ev, "printed": printed}
            return {"kind": "smc_events", "value": ev, "printed": printed}

        mark_lists = [v for v in namespace.values() if v is not ev and _looks_like_marks_list(v)]
        if mark_lists:
            combined = []
            for ml in mark_lists:
                combined.extend(ml)
            return {"kind": "marks", "value": combined, "printed": printed}

        # Last resort: the single most common mistake in scripts shared during
        # development of this feature was defining exactly one "do the work"
        # function and never actually calling it (or calling it but never
        # doing anything with what it returned) — the function itself was
        # always correct, only that one wiring step was missing. Rather than
        # just erroring, if there's exactly one unambiguous candidate, call it
        # with the candle data and make a best effort to use whatever it
        # returns. A script that already has a working result/events/run_smc
        # never reaches this point at all — this only ever fires when nothing
        # else produced anything.
        auto_fn = _find_auto_callable_function(namespace)
        if auto_fn is not None:
            try:
                auto_result = auto_fn(lower_df)
            except Exception as e:
                # "_retry_main": if the script has its own __main__ block, that
                # block may well call this function with the right inputs —
                # main() gives it a chance before settling on this error.
                return {
                    "error": f"Tried calling {auto_fn.__name__}(df) automatically since it "
                             f"was defined but never called, but it raised: {str(e)[:400]}",
                    "printed": printed,
                    "_retry_main": True,
                }
            interpreted = _interpret_auto_result(auto_result)
            if interpreted is not None:
                interpreted["printed"] = printed
                return interpreted
        return None

    # Scripts written as "define some functions, then do the actual work
    # under `if __name__ == "__main__":`" are the normal shape of a
    # standalone Python file — but the first run below uses the name
    # "finxview_script", so that block doesn't execute. That's deliberate
    # for the first pass: such blocks often load their own data or hit the
    # network (blocked here), and running them eagerly would break scripts
    # that already work through the other paths. It's only when nothing
    # else produced anything that the script gets run a second time AS
    # __main__, exactly the way its author intended it to be run.
    has_main_block = re.search(r"__name__\s*==\s*['\"]__main__['\"]", code) is not None

    def _memory_error_exit():
        _write_result(result_path, {
            "error": f"Script exceeded the {mem_mb}MB memory limit.",
            "printed": "".join(printed_parts),
        })
        sys.exit(0)

    try:
        namespace = _run_both_conventions(False)
    except MemoryError:
        _memory_error_exit()
    except Exception as e:
        msg = str(e)[:500] or e.__class__.__name__
        _write_result(result_path, {"error": msg, "printed": "".join(printed_parts)})
        sys.exit(0)  # runner itself succeeded — it's the USER script that raised

    first_printed = "".join(printed_parts)
    outcome = _detect(namespace, first_printed)
    retry_flag = bool(outcome.pop("_retry_main", False)) if isinstance(outcome, dict) else False

    if has_main_block and (outcome is None or retry_flag):
        printed_parts.clear()  # the second run re-executes the module-level code too
        main_ns, main_error, main_outcome = None, None, None
        try:
            main_ns = _run_both_conventions(True)
        except MemoryError:
            _memory_error_exit()
        except Exception as e:
            main_error = str(e)[:350] or e.__class__.__name__
        main_printed = "".join(printed_parts)
        if main_ns is not None:
            main_outcome = _detect(main_ns, main_printed)
            if isinstance(main_outcome, dict):
                main_outcome.pop("_retry_main", None)
                if "error" not in main_outcome:
                    _write_result(result_path, main_outcome)
                    return
        # The __main__ run didn't rescue it. Say what it did, so the failure
        # isn't a mystery — this is the part a user can actually act on.
        if main_error:
            main_note = f"it raised: {main_error}"
        elif main_outcome is None:
            main_note = "it ran, but still produced nothing to draw"
        else:
            main_note = main_outcome["error"][:300]
        base = outcome["error"] if isinstance(outcome, dict) else (
            "Script ran but produced nothing to draw. Set a "
            "result = {\"lines\": [...]} dict, define run_smc(df) returning "
            "(events, hashes), or just set events = [...] directly."
        )
        _write_result(result_path, {
            "error": f"{base}\n\n(Also tried running the script's `if __name__ == \"__main__\"` block: {main_note})",
            "printed": main_printed or first_printed,
        })
        return

    if outcome is not None:
        _write_result(result_path, outcome)
        return

    _write_result(result_path, {
        "error": (
            "Script ran but produced nothing to draw. Set a "
            "result = {\"lines\": [...]} dict, define run_smc(df) returning "
            "(events, hashes), or just set events = [...] directly."
        ),
        "printed": first_printed,
    })

if __name__ == "__main__":
    main()
