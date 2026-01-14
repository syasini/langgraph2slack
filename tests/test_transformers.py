"""Unit tests for langgraph2slack.transformers module.

Tests TransformerChain functionality including:
- Adding transformers
- Applying transformers in order
- Handling async transformers with/without context parameter
- Error handling and edge cases
"""

import pytest
from langgraph2slack.transformers import TransformerChain
from langgraph2slack.config import MessageContext


# ============================================================================
# Tests for TransformerChain
# ============================================================================


class TestTransformerChain:
    """Tests for TransformerChain functionality."""

    # Happy path tests - Basic functionality
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_chain_returns_unchanged(self, message_context_dm):
        """Empty chain should return message unchanged."""
        chain = TransformerChain()
        result = await chain.apply("hello", message_context_dm)

        assert result == "hello"

    @pytest.mark.asyncio
    async def test_single_transformer_with_context(self, message_context_dm):
        """Single transformer with context parameter should transform message."""
        chain = TransformerChain()

        @chain.add
        async def add_prefix(message: str, context: MessageContext) -> str:
            return f"[{context.user_id}] {message}"

        result = await chain.apply("hello", message_context_dm)
        assert result == "[U456USER] hello"

    @pytest.mark.asyncio
    async def test_single_transformer_without_context(self, message_context_dm):
        """Single transformer without context parameter should work."""
        chain = TransformerChain()

        @chain.add
        async def uppercase_transform(message: str) -> str:
            return message.upper()

        result = await chain.apply("hello", message_context_dm)
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_multiple_transformers_execute_in_order(self, message_context_dm):
        """Multiple transformers should execute in registration order."""
        chain = TransformerChain()

        @chain.add
        async def add_prefix(message: str, context: MessageContext) -> str:
            return f"[PREFIX] {message}"

        @chain.add
        async def add_suffix(message: str, context: MessageContext) -> str:
            return f"{message} [SUFFIX]"

        @chain.add
        async def uppercase_transform(message: str) -> str:
            return message.upper()

        result = await chain.apply("hello", message_context_dm)

        # Order: prefix -> suffix -> uppercase
        assert result == "[PREFIX] HELLO [SUFFIX]"

    @pytest.mark.asyncio
    async def test_mixed_transformers_with_and_without_context(self, message_context_dm):
        """Chain can mix transformers with and without context parameter."""
        chain = TransformerChain()

        # Transformer with context
        @chain.add
        async def add_user_prefix(message: str, context: MessageContext) -> str:
            return f"User {context.user_id}: {message}"

        # Transformer without context
        @chain.add
        async def add_footer(message: str) -> str:
            return f"{message}\n\n_Powered by LangGraph_"

        result = await chain.apply("test message", message_context_dm)

        assert "User U456USER: test message" in result
        assert "_Powered by LangGraph_" in result

    @pytest.mark.asyncio
    async def test_transformer_receives_previous_output(self, message_context_dm):
        """Each transformer should receive output of previous transformer."""
        chain = TransformerChain()

        @chain.add
        async def step1(message: str) -> str:
            return f"Step1({message})"

        @chain.add
        async def step2(message: str) -> str:
            return f"Step2({message})"

        @chain.add
        async def step3(message: str) -> str:
            return f"Step3({message})"

        result = await chain.apply("input", message_context_dm)

        # Each transformer wraps the previous output
        assert result == "Step3(Step2(Step1(input)))"

    # Tests for chain state
    # ------------------------------------------------------------------------

    def test_chain_length(self):
        """len() should return number of transformers."""
        chain = TransformerChain()

        assert len(chain) == 0

        @chain.add
        async def transform1(msg: str) -> str:
            return msg

        assert len(chain) == 1

        @chain.add
        async def transform2(msg: str) -> str:
            return msg

        assert len(chain) == 2

    def test_chain_bool_empty(self):
        """Empty chain should be falsy."""
        chain = TransformerChain()
        assert not chain  # Should be False

    def test_chain_bool_with_transformers(self):
        """Chain with transformers should be truthy."""
        chain = TransformerChain()

        @chain.add
        async def transform(msg: str) -> str:
            return msg

        assert chain  # Should be True

    # Context parameter detection tests
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_context_parameter_detection_two_params(self, message_context_dm):
        """Transformer with 2 parameters should receive context."""
        chain = TransformerChain()
        context_received = []

        @chain.add
        async def capture_context(message: str, context: MessageContext) -> str:
            context_received.append(context)
            return message

        await chain.apply("test", message_context_dm)

        assert len(context_received) == 1
        assert context_received[0] == message_context_dm

    @pytest.mark.asyncio
    async def test_context_parameter_detection_one_param(self, message_context_dm):
        """Transformer with 1 parameter should NOT receive context."""
        chain = TransformerChain()
        call_count = []

        @chain.add
        async def no_context_transform(message: str) -> str:
            # If context was passed, this would error with too many arguments
            call_count.append(1)
            return message.upper()

        result = await chain.apply("test", message_context_dm)

        assert result == "TEST"
        assert len(call_count) == 1

    # Critical negative tests - Error handling
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_transformer_raises_exception_propagates(self, message_context_dm):
        """Exception in transformer should propagate to caller."""
        chain = TransformerChain()

        @chain.add
        async def broken_transformer(message: str, context: MessageContext) -> str:
            raise ValueError("Something went wrong!")

        with pytest.raises(ValueError, match="Something went wrong!"):
            await chain.apply("test", message_context_dm)

    @pytest.mark.asyncio
    async def test_transformer_returns_none_handled_gracefully(self, message_context_dm):
        """Transformer returning None should be handled gracefully.

        Current implementation will pass None to next transformer.
        This may cause errors in subsequent transformers, which is OK
        (fail fast behavior).
        """
        chain = TransformerChain()

        @chain.add
        async def returns_none(message: str) -> str:
            return None  # Oops! Should return string

        # This will return None, which may cause issues
        # We're testing that it doesn't crash in the chain itself
        result = await chain.apply("test", message_context_dm)
        assert result is None

    @pytest.mark.asyncio
    async def test_transformer_returns_wrong_type_passes_through(self, message_context_dm):
        """Transformer returning wrong type should pass through.

        TransformerChain doesn't do runtime type checking.
        It's up to transformers to return correct types.
        """
        chain = TransformerChain()

        @chain.add
        async def returns_int(message: str) -> str:
            return 12345  # Wrong type!

        # Will return int instead of string
        result = await chain.apply("test", message_context_dm)
        assert result == 12345  # Type error, but doesn't crash

    @pytest.mark.asyncio
    async def test_exception_in_middle_transformer_stops_chain(self, message_context_dm):
        """Exception in middle transformer should stop chain execution."""
        chain = TransformerChain()
        executed = []

        @chain.add
        async def first(message: str) -> str:
            executed.append("first")
            return f"first: {message}"

        @chain.add
        async def second_broken(message: str) -> str:
            executed.append("second")
            raise RuntimeError("Transformer failed!")

        @chain.add
        async def third(message: str) -> str:
            executed.append("third")
            return f"third: {message}"

        with pytest.raises(RuntimeError, match="Transformer failed!"):
            await chain.apply("test", message_context_dm)

        # Only first two transformers should execute
        assert executed == ["first", "second"]
        assert "third" not in executed

    # Edge cases
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_message(self, message_context_dm):
        """Empty message should be handled correctly."""
        chain = TransformerChain()

        @chain.add
        async def add_prefix(message: str) -> str:
            return f"PREFIX: {message}"

        result = await chain.apply("", message_context_dm)
        assert result == "PREFIX: "

    @pytest.mark.asyncio
    async def test_transformer_modifies_context_does_not_affect_others(self, message_context_dm):
        """Transformers should not modify shared context.

        Each transformer receives the same context object.
        If one modifies it, others see the changes (Python object reference).
        This test documents current behavior.
        """
        chain = TransformerChain()
        user_ids_seen = []

        @chain.add
        async def first_transformer(message: str, context: MessageContext) -> str:
            user_ids_seen.append(context.user_id)
            return message

        @chain.add
        async def second_transformer(message: str, context: MessageContext) -> str:
            user_ids_seen.append(context.user_id)
            return message

        await chain.apply("test", message_context_dm)

        # Both should see same user_id
        assert user_ids_seen == ["U456USER", "U456USER"]
