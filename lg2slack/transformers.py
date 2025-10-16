"""Transformer chain management.

Handles input and output transformers that process messages before/after
sending to LangGraph.
"""

from typing import Callable, List, Awaitable, Union
from .config import MessageContext
import inspect

# Type alias for transformer functions
# Signature: async def transformer(message: str, context: MessageContext) -> str
# Or: async def transformer(message: str) -> str (context optional)
TransformerFunc = Union[
    Callable[[str, MessageContext], Awaitable[str]],
    Callable[[str], Awaitable[str]]
]


class TransformerChain:
    """Manages a chain of transformers.

    Transformers are async functions that take (message, context) and return
    a transformed message. Context parameter is optional.
    They're applied in the order they were registered.

    Example:
        chain = TransformerChain()

        @chain.add
        async def first(msg, ctx):
            return f"[1] {msg}"

        @chain.add
        async def second(msg):  # context optional
            return f"[2] {msg}"

        result = await chain.apply("hello", context)
        # result: "[2] [1] hello"
    """

    def __init__(self):
        """Initialize empty transformer chain."""
        self._transformers: List[TransformerFunc] = []

    def add(self, transformer: TransformerFunc) -> TransformerFunc:
        """Add a transformer to the chain.

        This method is designed to work as a decorator.

        Args:
            transformer: Async function (str, MessageContext) -> str
                        or (str) -> str (context optional)

        Returns:
            The transformer function (so it works as a decorator)

        Example:
            @chain.add
            async def my_transformer(message, context):
                return message.upper()

            @chain.add
            async def simple_transformer(message):
                return message.lower()
        """
        self._transformers.append(transformer)
        return transformer

    async def apply(self, message: str, context: MessageContext) -> str:
        """Apply all transformers in order.

        Each transformer receives the output of the previous one.
        If no transformers are registered, returns the message unchanged.
        Automatically detects if transformer accepts context parameter.

        Args:
            message: Input message
            context: Message context

        Returns:
            Transformed message after applying all transformers

        Example:
            # If we have transformers: [add_prefix, add_suffix]
            # Input: "hello"
            # After add_prefix: "PREFIX: hello"
            # After add_suffix: "PREFIX: hello :SUFFIX"
        """
        # Start with the original message
        result = message

        # Apply each transformer in order
        for transformer in self._transformers:
            # Check if transformer accepts context parameter
            sig = inspect.signature(transformer)
            param_count = len(sig.parameters)

            if param_count >= 2:
                # Transformer accepts (message, context)
                result = await transformer(result, context)
            else:
                # Transformer only accepts (message)
                result = await transformer(result)

        return result

    def __len__(self) -> int:
        """Get number of transformers in chain."""
        return len(self._transformers)

    def __bool__(self) -> bool:
        """True if chain has at least one transformer."""
        return len(self._transformers) > 0
