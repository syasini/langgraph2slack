# Code Enhancements Tracker

This file tracks refactoring opportunities and code elegance improvements.

---

## 💡 High Value Enhancements

### ✅ Opportunity #1: Refactor _normalize_reactions()
**Status:** Pending
**Branch:** `claude/refactor-normalize-reactions-011CUomKh1oKXEihWTyV7Sra`
**Priority:** HIGH (for code elegance)
**Files Affected:** `bot.py` (lines 328-407)

**Problem:**
- Method is 79 lines long
- Does multiple things: backward compatibility, validation, defaults, normalization
- Hard to test individual pieces
- Single responsibility principle violation

**Solution:**
Break into smaller methods:
- `_convert_legacy_reaction()` - Handle backward compatibility
- `_normalize_single_reaction()` - Validate & normalize one reaction
- `_validate_reaction_fields()` - Validate required fields
- `_normalize_reactions()` - Orchestrate the above

**Expected Impact:**
- Each method has single responsibility
- Easier to test individually
- More readable
- Easier to modify

**Estimated Time:** 1-2 hours

---

## 💡 Medium Value Enhancements

### ✅ Opportunity #2: Simplify Buffer Flush Logic
**Status:** ✅ Complete (commit: b964c96)
**Branch:** `claude/simplify-buffer-flush-011CUomKh1oKXEihWTyV7Sra`
**Priority:** MEDIUM
**Files Affected:** `handlers/stream.py` (lines 361-365)

**Problem:**
- Pattern "check → flush → update time" is manual
- Easy to forget updating `last_flush_time`
- No visibility into why buffer flushed

**Solution:**
```python
def _should_flush_buffer(self, buffer, last_flush_time):
    """Check if should flush. Returns (should_flush, reason)."""
    if not buffer:
        return False, None

    time_elapsed = time.time() - last_flush_time

    if time_elapsed >= self.stream_buffer_time:
        return True, f"time ({time_elapsed:.2f}s)"
    if len(buffer) >= self.stream_buffer_max_chunks:
        return True, f"chunks ({len(buffer)})"

    return False, None

async def _flush_buffer_and_return_time(self, buffer, channel_id, stream_ts):
    """Flush buffer and return current time."""
    await self._flush_buffer(buffer, channel_id, stream_ts)
    return time.time()
```

**Expected Impact:**
- Impossible to forget updating time
- Better logging (shows flush reason)
- More testable

**Estimated Time:** 30 minutes

---

### ✅ Opportunity #3: Add Configuration Validation
**Status:** ✅ Complete (commit: ac95067)
**Branch:** `claude/add-config-validation-011CUomKh1oKXEihWTyV7Sra`
**Priority:** MEDIUM
**Files Affected:** `bot.py` (`__init__` method)

**Problem:**
- No validation of numeric parameters
- User could pass negative or unreasonable values
- Silent failures or unexpected behavior

**Solution:**
```python
def _validate_buffer_config(self, buffer_time, max_chunks):
    """Validate buffer configuration parameters."""
    if buffer_time <= 0:
        raise ValueError(f"stream_buffer_time must be positive, got {buffer_time}")
    if buffer_time > 5.0:
        logger.warning(
            f"stream_buffer_time={buffer_time}s is very high. "
            "Recommended range: 0.05-0.2 seconds."
        )

    if max_chunks < 1:
        raise ValueError(f"stream_buffer_max_chunks must be >= 1, got {max_chunks}")
    if max_chunks > 100:
        logger.warning(
            f"stream_buffer_max_chunks={max_chunks} is very high."
        )
```

**Expected Impact:**
- Catches configuration errors early
- Provides helpful warnings
- Better user experience

**Estimated Time:** 30 minutes

---

## 📋 Checklist

- [x] Opportunity #1: Refactor _normalize_reactions()
- [x] Opportunity #2: Simplify buffer flush logic
- [x] Opportunity #3: Add configuration validation

---

## Notes

- Lower priority than fixes.md items
- Can be done incrementally
- Focus on readability and maintainability
- Each enhancement should have tests
