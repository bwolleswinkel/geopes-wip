"""Script to test the whether having the config class in a separate file works fine"""

import traceback, inspect
from typing import Literal, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings

# FROM: GitHub Copilot GPT-4.1 | 2026/01/30 [untested/unverified]
if TYPE_CHECKING:
    from testPolytopeOnConvert import Polytope


@dataclass
class CFG:
    _ON_POLY_CONVERT: Literal['allow', 'warning', 'error'] = field(default='allow', init=False)


    @staticmethod
    def retrieve_stack() -> str:
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
    def on_convert(poly: Polytope | None = None) -> bool:
        """Whether to raise an error when converting between polytope representations. Optionally include polytope context."""
        match cfg._ON_POLY_CONVERT:
            case 'allow':
                return True
            case 'warning' | 'error':
                poly_str = f"Caught the conversion of {poly.__repr__()} from one representation to another. " if poly is not None else "Conversion between polytope representations detected. "
                msg = f"{poly_str}Note that 'CFG._ON_POLY_CONVERT' is set to '{cfg._ON_POLY_CONVERT}'.\n\nCall chain:\n{CFG.retrieve_stack()}\n"
                match cfg._ON_POLY_CONVERT:
                    case 'warning':
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    case 'error':
                        raise RuntimeError(f"\033[95m{msg}\033[0m")
                    case _:
                        raise ValueError(f"Unrecognized configuration for on_convert '{cfg._ON_POLY_CONVERT}'")
                return True
            case _:
                raise ValueError(f"Unrecognized configuration for on_convert '{cfg._ON_POLY_CONVERT}'")
            
    @staticmethod
    def set_on_poly_convert(mode: Literal['allow', 'warning', 'error']) -> None:
        """Set the behavior for converting between polytope representations"""
        if mode not in ['allow', 'warning', 'error']:
            raise ValueError(f"Unrecognized mode '{mode}' for on_convert. Must be one of 'allow', 'warning', 'error'.")
        cfg._ON_POLY_CONVERT = mode

global cfg
cfg = CFG()