# Copyright 2026 Open-MOPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep vLLM's in-place Transformers class decoration away from training models."""

import importlib
import inspect
import sys
from contextlib import ExitStack, contextmanager
from itertools import count
from unittest.mock import patch


_class_ids = count()


def _resolve_wrapped_function(module_name, qualname, depth):
    """Resolve a decorator's inner function in this process, including on AOT load."""
    function = importlib.import_module(module_name)
    for part in qualname.split("."):
        function = getattr(function, part)
    for _ in range(depth):
        function = function.__wrapped__
    return function


@contextmanager
def _serialize_wrapped_functions():
    """Preserve functools.wraps closures when PyTorch serializes an AOT artifact."""
    try:
        from torch._dynamo.aot_compile import AOTCompilePickler
    except ModuleNotFoundError as exc:
        if exc.name != "torch._dynamo.aot_compile":
            raise
        yield
        return

    original_reducer = AOTCompilePickler.reducer_override

    def reducer(self, obj):
        if inspect.isfunction(obj) and "<locals>" not in obj.__qualname__:
            # functools.wraps gives inner and outer functions the same name.
            # Pickle's global lookup finds only the outer function. Store the
            # inner function's position instead of changing its name or code.
            try:
                public_function = _resolve_wrapped_function(obj.__module__, obj.__qualname__, 0)
            except (ImportError, AttributeError):
                public_function = None
            depth = 0
            seen = set()
            while public_function is not None and id(public_function) not in seen:
                if public_function is obj:
                    if depth:
                        return _resolve_wrapped_function, (obj.__module__, obj.__qualname__, depth)
                    break
                seen.add(id(public_function))
                public_function = getattr(public_function, "__wrapped__", None)
                depth += 1
        return original_reducer(self, obj)

    with patch.object(AOTCompilePickler, "reducer_override", reducer):
        yield


@contextmanager
def isolate_transformers_compile():
    """Scope HF constructor redirects to synchronous, colocated vLLM initialization.

    Recent vLLM versions mutate a decoder class's bases, __init__, and __call__.
    Actor/teacher instances of that same HF class then enter vLLM's __call__
    without its compile state. Decorate a private subclass instead, preserving
    the original class and PyTorch's training hooks.
    """
    try:
        from vllm.model_executor.models.transformers.base import Base
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "vllm.model_executor.models.transformers",
            "vllm.model_executor.models.transformers.base",
        }:
            raise
        yield
        return

    if not hasattr(Base, "_decorate_cls_for_torch_compile"):
        yield
        return

    original_decorate = Base._decorate_cls_for_torch_compile
    isolated_classes = {}
    with ExitStack() as stack:
        stack.enter_context(_serialize_wrapped_functions())

        def decorate(self, cls, dynamic_arg_dims, enable_if, is_encoder):
            if cls in isolated_classes.values():
                isolated_cls = cls
            elif cls in isolated_classes:
                isolated_cls = isolated_classes[cls]
            else:
                # Retain __name__ for Transformers' model conversion registry.
                # A stable module binding also makes the subclass serializable.
                alias = f"_Vllm_{cls.__name__}_{next(_class_ids)}"
                isolated_cls = type(cls.__name__, (cls,), {"__module__": __name__, "__qualname__": alias})
                globals()[alias] = isolated_cls
                isolated_classes[cls] = isolated_cls

                # AutoModel may already have cached the original class. Redirect
                # its classmethod factory without modifying the auto registry.
                original_from_config = cls._from_config.__func__

                def from_config(factory_cls, *args, **kwargs):
                    target_cls = isolated_cls if factory_cls is cls else factory_cls
                    return original_from_config(target_cls, *args, **kwargs)

                stack.enter_context(patch.object(cls, "_from_config", classmethod(from_config)))

                # Composite HF models construct decoders via module globals.
                # Populate public lazy exports first, so AutoModel cannot cache
                # the temporary private class in a Transformers package.
                for module_name in (cls.__module__.rpartition(".")[0], "transformers"):
                    getattr(importlib.import_module(module_name), cls.__name__, None)
                module = sys.modules[cls.__module__]
                for name, value in list(vars(module).items()):
                    if value is cls:
                        stack.enter_context(patch.object(module, name, isolated_cls))

            return original_decorate(self, isolated_cls, dynamic_arg_dims, enable_if, is_encoder)

        stack.enter_context(patch.object(Base, "_decorate_cls_for_torch_compile", decorate))
        yield
