"""Script to test the whether having the config class in a separate file works fine"""

import traceback, inspect
from typing import Literal, TYPE_CHECKING
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings

# FROM: GitHub Copilot GPT-4.1 | 2026/01/30 [untested/unverified]
if TYPE_CHECKING:
    from testPolytopeOnConvert import Polytope
    

@dataclass
class GlobalConfig:
    """CAUTION: This class should be instantiated only once in the module `global.py` and that instance imported elsewhere. This class maintains a global configuration for the package."""
    _on_poly_convert: Literal['allow', 'warning', 'error'] = field(default='allow', init=False)
    on_poly_assign: Literal['reduce', 'leave'] = field(default='reduce', init=False)

    # TODO: It would be really nice the also print [function] or [property] or [method] next to the function names where possible. I've asked GPT-4.1 to do this but it fails miserably. Maybe try again later with Clause Sonnet 3
    @staticmethod
    def _retrieve_stack() -> str:
        """Return a formatted call chain with class and method/property names where possible, one per line, with aligned columns: right-aligned line numbers, left-aligned function names, left-aligned script names. The last entry (CFG.retrieve_stack) is omitted."""
        # FROM: GitHub Copilot GPT-4.1 | 2026/01/30 [untested/unverified]
        stack = traceback.extract_stack()
        # Remove the last entry (CFG.retrieve_stack)
        stack = stack[:-1]
        current_frames = inspect.stack()
        frame_map = {}
        for frame_info in current_frames:
            frame = frame_info.frame
            code = frame.f_code
            func_name = code.co_name
            filename = code.co_filename
            lineno = frame_info.lineno
            cls_name = None
            if 'self' in frame.f_locals:
                cls_name = type(frame.f_locals['self']).__name__
            elif 'cls' in frame.f_locals:
                cls_name = frame.f_locals['cls'].__name__
            if cls_name is None:
                qualname = code.co_qualname if hasattr(code, 'co_qualname') else None
                if qualname and '.' in qualname:
                    cls_name = qualname.split('.')[0]
            frame_map[(filename, func_name, lineno)] = cls_name
        # Prepare for alignment
        filtered_stack = [frame for frame in stack if "site-packages" not in frame.filename and "lib/python" not in frame.filename]
        # Collect all fields for alignment
        stack_entries = []
        for frame in filtered_stack:
            cls_name = None
            for (fname, func, lno), cname in frame_map.items():
                if fname == frame.filename and func == frame.name:
                    cls_name = cname
                    break
            if cls_name:
                func = f"{cls_name}.{frame.name}"
            else:
                func = frame.name
            script = frame.filename.split('/')[-1]
            stack_entries.append((frame.lineno, func, script))
        # Find max widths for alignment
        max_lineno_len = max((len(str(lineno)) for lineno, _, _ in stack_entries), default=2)
        max_func_len = max((len(func) for _, func, _ in stack_entries), default=8)
        max_script_len = max((len(script) for _, _, script in stack_entries), default=8)
        formatted_stack = []
        for lineno, func, script in stack_entries:
            lineno_fmt = f"{lineno:>{max_lineno_len}}"
            func_fmt = f"{func:<{max_func_len}}"
            script_fmt = f"({script})".ljust(max_script_len + 2)
            formatted_stack.append(f"  {lineno_fmt}: {func_fmt}  {script_fmt}")
        return "\n".join(formatted_stack)

    @staticmethod
    def on_poly_convert(poly: Polytope | None = None) -> bool:
        """Whether to raise an error when converting between polytope representations. Optionally include polytope context."""
        match CFG._on_poly_convert:
            case 'allow':
                return True
            case 'warning' | 'error':
                poly_str = f"Caught the conversion of {poly.__repr__()} from one representation to another. " if poly is not None else "Conversion between polytope representations detected. "
                msg = f"{poly_str}Note that 'CFG._ON_POLY_CONVERT' is set to '{CFG._on_poly_convert}'.\n\nCall chain:\n{GlobalConfig._retrieve_stack()}\n"
                match CFG._on_poly_convert:
                    case 'warning':
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    case 'error':
                        raise RuntimeError(f"\033[95m{msg}\033[0m")
                    case _:
                        raise ValueError(f"Unrecognized configuration for on_convert '{CFG._on_poly_convert}'")
                return True
            case _:
                raise ValueError(f"Unrecognized configuration for on_convert '{CFG._on_poly_convert}'")
            
    @contextmanager
    def poly_convert_context(self, mode: Literal['allow', 'warning', 'error']):
        """Context manager to temporarily set the on_poly_convert behavior"""
        old_mode = self._on_poly_convert
        try:
            self._on_poly_convert = mode
            yield
        finally:
            self._on_poly_convert = old_mode
            
    
CFG = GlobalConfig()


def set_on_poly_convert(mode: Literal['allow', 'warning', 'error']) -> None:
    """Set the behavior when converting between polytope representations"""
    if mode not in ['allow', 'warning', 'error']:
        raise ValueError(f"Unrecognized mode '{mode}' for on_convert. Must be one of 'allow', 'warning', 'error'.")
    CFG._on_poly_convert = mode


def on_poly_convert(mode: Literal['allow', 'warning', 'error']) -> contextmanager:
    """Get the current behavior when converting between polytope representations"""
    return CFG.poly_convert_context(mode)


# TODO: Also check out the implementation below, on how to make this setting/getting of variables super robust!
'''
# FROM: Google Gemini 3 | 2026/01/30
import contextlib

@dataclass
class ConfigSchema:
    """This defines exactly what settings are allowed and their defaults."""
    precision: int = 2
    verbose: bool = False

class GlobalConfig:
    def __init__(self):
        # Initializing the internal data store
        self.__dict__['_settings'] = {
            "precision": 2,
            "verbose": False,
            "theme": "dark"
        }
        self.__dict__['_locked'] = True # The gate is closed

    def __getattr__(self, name):
        """Allows config.precision to return self._settings['precision']"""
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Prevents direct modification."""
        if getattr(self, '_locked', False):
            raise AttributeError(
                f"Cannot set '{name}' directly. Use 'set_options()' instead."
            )
        super().__setattr__(name, value)

    def _update(self, **kwargs):
        """Internal bridge to update the locked dict."""
        self.__dict__['_locked'] = False
        try:
            for k, v in kwargs.items():
                if k in self._settings: self._settings[k] = v
        finally:
            self.__dict__['_locked'] = True

    @contextlib.contextmanager
    def context(self, **kwargs):
        """Temporary overrides."""
        old_state = self._settings.copy()
        try:
            self.set_options(**kwargs)
            yield
        finally:
            self.set_options(**old_state)

# Instantiate the singleton for the module
_config = GlobalConfig()

# --- Public API functions defined right here ---

def set_algo_options(precision: int = None, verbose: bool = None):
    """Update algorithmic global state."""
    updates = {k: v for k, v in locals().items() if v is not None}
    _config._update(**updates)

def get_config():
    """Returns the current state for inspection."""
    return _config

# my_package/__init__.py
from .config import set_algo_options, get_config

__all__ = ["set_algo_options", "get_config"]
'''