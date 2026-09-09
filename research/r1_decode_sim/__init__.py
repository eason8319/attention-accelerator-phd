"""R1 解码模拟器组件，仅依赖 Python 标准库。"""

from .inputs import InputBundle, InputValidationError, load_inputs

__all__ = ["InputBundle", "InputValidationError", "load_inputs"]
