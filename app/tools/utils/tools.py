from functools import wraps
import inspect
from typing import Callable, get_type_hints, Any, Dict, get_origin, get_args, Union
import docstring_parser
from typing import Literal


def format_type_name(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        return "Union[" + ", ".join(format_type_name(arg) for arg in args) + "]"
    elif origin is list:
        args = get_args(tp)
        return f"List[{format_type_name(args[0])}]" if args else "List"
    elif origin is dict:
        args = get_args(tp)
        if args and len(args) == 2:
            return f"Dict[{format_type_name(args[0])}, {format_type_name(args[1])}]"
        return "Dict"
    elif hasattr(tp, '__name__'):
        return tp.__name__
    else:
        return str(tp)


def tool(description: str = None):
    """Decorator to mark methods as tools and generate a signature string and OpenAI tool schema."""
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        doc = docstring_parser.parse(docstring)

        # Extract parameter descriptions from docstring
        param_docs = {param.arg_name: param.description for param in doc.params}

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Determine if 'self' should be skipped
        parameters = list(sig.parameters.items())
        skip_self = parameters and parameters[0][0] == 'self'

        wrapper.__args__ = [
            name for name, _ in parameters if not (skip_self and name == 'self')
        ]

        type_hints = get_type_hints(func)

        # Build string schema
        arg_strs = []
        for param_name in wrapper.__args__:
            desc = param_docs.get(param_name, param_name)
            param_type = type_hints.get(param_name, str)
            type_name = format_type_name(param_type)
            arg_strs.append(f'{param_name}: {type_name}="{desc}"')

        func_description = description or doc.short_description or f"Execute {func.__name__}"
        formatted_str = f'{func.__name__}({", ".join(arg_strs)}) - {func_description}'

        wrapper._tool_schema = formatted_str
        wrapper._is_tool = True

        # Build OpenAI-compatible function calling schema
        properties: Dict[str, Any] = {}
        required = []

        for param_name in wrapper.__args__:
            param = sig.parameters[param_name]
            param_type = type_hints.get(param_name, str)
            schema = python_type(param_type)
            schema["description"] = param_docs.get(param_name, param_name)
            properties[param_name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        wrapper._openai_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

        return wrapper

    return decorator


def python_type(tp: Any) -> Dict[str, Any]:
    """Convert Python type annotation"""
    origin = get_origin(tp)

    if origin is Literal:
        # Enum support
        args = get_args(tp)
        return {
            "type": "string",
            "enum": list(args)
        }

    if origin is Union:
        args = get_args(tp)
        # Handle Optional[X] == Union[X, NoneType]
        non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none_args) == 1:
            # It's Optional
            schema = python_type(non_none_args[0])
            schema["nullable"] = True
            return schema
        else:
            # More complex Union - map to multiple types
            types = []
            for arg in args:
                if arg is type(None):
                    continue
                types.append(python_type(arg).get("type", "string"))
            return {
                "anyOf": [{"type": t} for t in types],
                "nullable": type(None) in args
            }

    if tp == str:
        return {"type": "string"}
    if tp == int:
        return {"type": "integer"}
    if tp == float:
        return {"type": "number"}
    if tp == bool:
        return {"type": "boolean"}
    if origin == list:
        args = get_args(tp)
        item_type = python_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}
    if origin == dict:
        args = get_args(tp)
        value_type = python_type(args[1]) if len(args) > 1 else {"type": "string"}
        return {"type": "object", "additionalProperties": value_type}

    return {"type": "string"}  # default fallback
