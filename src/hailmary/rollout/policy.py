"""Frozen downstream-policy protocol and mutation audits."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import dis
import enum
import functools
import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from hailmary.ids import canonical_data, canonical_json, content_hash


@runtime_checkable
class FrozenPolicy(Protocol):
    """A rollout policy may select actions but must not learn or mutate."""

    def select_action(self, context: Any) -> Any | None: ...


@dataclass(frozen=True)
class NoOpPolicy:
    """Stateless policy useful for fixed-horizon and inner comparisons."""

    action: Any | None = None

    def select_action(self, context: Any) -> Any | None:
        del context
        return self.action


def _explicit_fingerprint_error(path: str, value: Any) -> TypeError:
    return TypeError(
        f"{path} contains non-canonical behavior state "
        f"{type(value).__module__}.{type(value).__qualname__}; "
        "use a policy object implementing explicit policy_fingerprint()"
    )


def _callable_reference(value: Any) -> dict[str, Any]:
    target = getattr(value, "__func__", value)
    reference: dict[str, Any] = {
        "module": getattr(target, "__module__", type(target).__module__),
        "qualname": getattr(target, "__qualname__", type(target).__qualname__),
    }
    code = getattr(target, "__code__", None)
    if isinstance(code, types.CodeType):
        reference["code"] = _canonical_code_state(code)
    return reference


def _canonical_code_constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"__code__": _canonical_code_state(value)}
    if value is Ellipsis:
        return {"__singleton__": "Ellipsis"}
    if value is NotImplemented:
        return {"__singleton__": "NotImplemented"}
    if isinstance(value, complex):
        return {
            "__complex__": [
                canonical_data(value.real),
                canonical_data(value.imag),
            ]
        }
    if isinstance(value, tuple):
        return [_canonical_code_constant(item) for item in value]
    if isinstance(value, frozenset):
        converted = [_canonical_code_constant(item) for item in value]
        return sorted(converted, key=canonical_json)
    try:
        return canonical_data(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "callable frozen-policy code contains an unsupported constant"
        ) from exc


def _canonical_code_state(code: types.CodeType) -> dict[str, Any]:
    return {
        "bytecode": code.co_code,
        "constants": [_canonical_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "exceptiontable": getattr(code, "co_exceptiontable", b""),
    }


def _loaded_global_names(code: types.CodeType) -> tuple[str, ...]:
    names: set[str] = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in {
            "LOAD_GLOBAL",
            "LOAD_NAME",
            "LOAD_FROM_DICT_OR_GLOBALS",
        } and isinstance(instruction.argval, str):
            names.add(instruction.argval)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_loaded_global_names(constant))
    return tuple(sorted(names))


def _canonical_class_behavior_state(
    instance_type: type,
    *,
    path: str,
    seen: set[int],
) -> list[dict[str, Any]]:
    """Capture user-class data and helper behavior reachable through self."""

    owners: list[dict[str, Any]] = []
    for owner in instance_type.__mro__:
        if owner is object or owner.__module__ == "builtins":
            continue
        identifier = id(owner)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, owner)
        seen.add(identifier)
        try:
            attributes: dict[str, Any] = {}
            for name, item in sorted(vars(owner).items()):
                if name.startswith("__"):
                    continue
                attribute_path = f"{path} class attribute {name!r}"
                if isinstance(
                    item, (types.MemberDescriptorType, types.GetSetDescriptorType)
                ):
                    continue
                if isinstance(item, (staticmethod, classmethod)):
                    attributes[name] = {
                        "descriptor": type(item).__name__,
                        "callable": _canonical_callable_state(
                            item.__func__,
                            path=attribute_path,
                            seen=seen,
                        ),
                    }
                    continue
                if isinstance(item, property):
                    attributes[name] = {
                        "descriptor": "property",
                        "getter": (
                            None
                            if item.fget is None
                            else _canonical_callable_state(
                                item.fget,
                                path=f"{attribute_path} getter",
                                seen=seen,
                            )
                        ),
                        "setter": (
                            None
                            if item.fset is None
                            else _canonical_callable_state(
                                item.fset,
                                path=f"{attribute_path} setter",
                                seen=seen,
                            )
                        ),
                        "deleter": (
                            None
                            if item.fdel is None
                            else _canonical_callable_state(
                                item.fdel,
                                path=f"{attribute_path} deleter",
                                seen=seen,
                            )
                        ),
                    }
                    continue
                if inspect.isroutine(item):
                    attributes[name] = {
                        "callable": _canonical_callable_state(
                            item,
                            path=attribute_path,
                            seen=seen,
                        )
                    }
                    continue
                if hasattr(item, "__get__"):
                    raise _explicit_fingerprint_error(attribute_path, item)
                attributes[name] = _canonical_behavior_value(
                    item,
                    path=attribute_path,
                    seen=seen,
                )
            owners.append(
                {
                    "type": _callable_reference(owner),
                    "attributes": attributes,
                }
            )
        finally:
            seen.remove(identifier)
    return owners


def _canonical_behavior_value(
    value: Any,
    *,
    path: str,
    seen: set[int],
    expand_callable: bool = True,
) -> Any:
    if inspect.ismodule(value):
        raise _explicit_fingerprint_error(path, value)
    if isinstance(value, type):
        if value.__module__ != "builtins":
            raise _explicit_fingerprint_error(path, value)
        return {"__type_reference__": _callable_reference(value)}
    if isinstance(value, functools.partial):
        identifier = id(value)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, value)
        seen.add(identifier)
        try:
            return {
                "__partial__": {
                    "function": _canonical_behavior_value(
                        value.func,
                        path=f"{path}.func",
                        seen=seen,
                    ),
                    "args": _canonical_behavior_value(
                        value.args,
                        path=f"{path}.args",
                        seen=seen,
                    ),
                    "keywords": _canonical_behavior_value(
                        value.keywords or {},
                        path=f"{path}.keywords",
                        seen=seen,
                    ),
                }
            }
        finally:
            seen.remove(identifier)
    if expand_callable and callable(value):
        return {
            "__callable__": _canonical_callable_state(
                value,
                path=path,
                seen=seen,
            )
        }

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        identifier = id(value)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, value)
        seen.add(identifier)
        try:
            return {
                "__dataclass_instance__": {
                    "type": _callable_reference(type(value)),
                    "class_behavior": _canonical_class_behavior_state(
                        type(value),
                        path=f"{path} class behavior",
                        seen=seen,
                    ),
                    "state": {
                        field.name: _canonical_behavior_value(
                            getattr(value, field.name),
                            path=f"{path}.{field.name}",
                            seen=seen,
                        )
                        for field in dataclasses.fields(value)
                    },
                }
            }
        finally:
            seen.remove(identifier)
    if isinstance(value, enum.Enum):
        return {
            "__enum__": {
                "type": _callable_reference(type(value)),
                "value": _canonical_behavior_value(
                    value.value,
                    path=f"{path}.value",
                    seen=seen,
                ),
            }
        }
    if isinstance(value, Mapping):
        identifier = id(value)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, value)
        seen.add(identifier)
        try:
            pairs = sorted(
                ((str(key), item) for key, item in value.items()),
                key=lambda pair: pair[0],
            )
            return {
                key: _canonical_behavior_value(
                    item,
                    path=f"{path}[{key!r}]",
                    seen=seen,
                )
                for key, item in pairs
            }
        finally:
            seen.remove(identifier)
    if isinstance(value, (list, tuple)):
        identifier = id(value)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, value)
        seen.add(identifier)
        try:
            return [
                _canonical_behavior_value(
                    item,
                    path=f"{path}[{index}]",
                    seen=seen,
                )
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(identifier)
    if isinstance(value, (set, frozenset)):
        identifier = id(value)
        if identifier in seen:
            raise _explicit_fingerprint_error(path, value)
        seen.add(identifier)
        try:
            converted = [
                _canonical_behavior_value(
                    item,
                    path=f"{path} set item",
                    seen=seen,
                )
                for item in value
            ]
            return sorted(converted, key=canonical_json)
        finally:
            seen.remove(identifier)

    try:
        return canonical_data(value)
    except (TypeError, RecursionError):
        pass

    identifier = id(value)
    if identifier in seen:
        raise _explicit_fingerprint_error(path, value)
    seen.add(identifier)
    try:
        state: dict[str, Any] = {}
        state_is_observable = False
        try:
            state.update(vars(value))
        except TypeError:
            pass
        else:
            state_is_observable = True
        for owner in type(value).__mro__:
            raw_slots = owner.__dict__.get("__slots__", ())
            slots = (raw_slots,) if isinstance(raw_slots, str) else tuple(raw_slots)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"} or slot in state:
                    continue
                try:
                    state[slot] = getattr(value, slot)
                except AttributeError:
                    continue
                state_is_observable = True
        if not state_is_observable:
            raise _explicit_fingerprint_error(path, value)
        return {
            "__instance__": {
                "type": _callable_reference(type(value)),
                "class_behavior": _canonical_class_behavior_state(
                    type(value),
                    path=f"{path} class behavior",
                    seen=seen,
                ),
                "state": _canonical_behavior_value(
                    state,
                    path=f"{path} instance state",
                    seen=seen,
                ),
            }
        }
    except (TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, TypeError) and "explicit policy_fingerprint" in str(exc):
            raise
        raise _explicit_fingerprint_error(path, value) from exc
    finally:
        seen.remove(identifier)


def _resolve_callable(selector: Callable[..., Any]) -> tuple[Any, Any | None]:
    if inspect.ismethod(selector) or inspect.isbuiltin(selector):
        return getattr(selector, "__func__", selector), getattr(
            selector,
            "__self__",
            None,
        )
    if inspect.isfunction(selector):
        return selector, None
    if not callable(selector):
        raise TypeError("callable frozen policy requires a callable selector")
    call = getattr(selector, "__call__")
    return getattr(call, "__func__", call), selector


def _canonical_callable_state(
    selector: Callable[..., Any],
    *,
    path: str = "callable",
    seen: set[int] | None = None,
) -> dict[str, Any]:
    """Capture code plus canonical closure, global, and bound-instance state."""

    function, bound_instance = _resolve_callable(selector)
    active = set() if seen is None else seen
    identifier = id(function)
    if identifier in active:
        return {"recursive_reference": _callable_reference(function)}
    active.add(identifier)
    try:
        return _canonical_resolved_callable_state(
            function,
            bound_instance,
            path=path,
            seen=active,
        )
    finally:
        active.remove(identifier)


def _canonical_resolved_callable_state(
    function: Any,
    bound_instance: Any | None,
    *,
    path: str,
    seen: set[int],
) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    closure_values: list[tuple[str, Any]] = []
    closure_names = tuple(getattr(code, "co_freevars", ()))
    for index, cell in enumerate(getattr(function, "__closure__", ()) or ()):
        name = closure_names[index] if index < len(closure_names) else str(index)
        try:
            value = cell.cell_contents
        except ValueError:
            value = {"empty_cell": True}
        closure_values.append(
            (
                name,
                _canonical_behavior_value(
                    value,
                    path=f"{path} closure {name!r}",
                    seen=seen,
                ),
            )
        )
    defaults = _canonical_behavior_value(
        getattr(function, "__defaults__", None),
        path=f"{path} defaults",
        seen=seen,
    )
    keyword_defaults = _canonical_behavior_value(
        getattr(function, "__kwdefaults__", None),
        path=f"{path} keyword defaults",
        seen=seen,
    )
    global_values: dict[str, Any] = {}
    function_globals = getattr(function, "__globals__", {})
    if isinstance(code, types.CodeType) and isinstance(function_globals, Mapping):
        for name in _loaded_global_names(code):
            if name in function_globals:
                global_values[name] = _canonical_behavior_value(
                    function_globals[name],
                    path=f"{path} global {name!r}",
                    seen=seen,
                )
    return {
        "module": getattr(function, "__module__", ""),
        "qualname": getattr(function, "__qualname__", type(function).__qualname__),
        "code": None if code is None else _canonical_code_state(code),
        "defaults": defaults,
        "keyword_defaults": keyword_defaults,
        "closure": closure_values,
        "globals": global_values,
        "bound_instance": (
            None
            if bound_instance is None
            else _canonical_behavior_value(
                bound_instance,
                path=f"{path} bound instance",
                seen=seen,
                expand_callable=False,
            )
        ),
    }


@dataclass(frozen=True)
class CallableFrozenPolicy:
    """Explicitly mark a side-effect-free callable as a frozen policy."""

    selector: Callable[[Any], Any | None]
    name: str | None = None

    def select_action(self, context: Any) -> Any | None:
        return self.selector(context)

    def policy_fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "callable_frozen_policy",
                "name": self.name,
                "selector": _canonical_behavior_value(
                    self.selector,
                    path="callable frozen-policy selector",
                    seen=set(),
                ),
            },
            namespace="hailmary.policy",
        )


def policy_fingerprint(policy: Any) -> str:
    """Fingerprint visible policy state before and after temporary rollouts."""

    explicit = getattr(policy, "policy_fingerprint", None)
    if callable(explicit):
        result = explicit()
        if not isinstance(result, str) or not result:
            raise TypeError("policy_fingerprint() must return a non-empty string")
        return result

    if callable(policy):
        payload = _canonical_behavior_value(
            policy,
            path="callable policy",
            seen=set(),
        )
        return content_hash(payload, namespace="hailmary.policy")
    selector = getattr(policy, "select_action", None)
    if callable(selector):
        return content_hash(
            {
                "kind": "frozen_policy_object",
                "type": _callable_reference(type(policy)),
                "selector": _canonical_behavior_value(
                    selector,
                    path="policy select_action",
                    seen=set(),
                ),
            },
            namespace="hailmary.policy",
        )
    return content_hash(
        _canonical_behavior_value(policy, path="policy", seen=set()),
        namespace="hailmary.policy",
    )


__all__ = [
    "CallableFrozenPolicy",
    "FrozenPolicy",
    "NoOpPolicy",
    "policy_fingerprint",
]
