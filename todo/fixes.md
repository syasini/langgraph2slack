# Code Fixes Tracker

This file tracks critical bugs and code smells that need fixing.

---

## 🔴 Critical Priority

### ✅ Issue #1: Duplicate Reaction Methods
**Status:** ✅ Complete (commit: b8f4223)
**Branch:** `claude/refactor-reaction-mixin-011CUomKh1oKXEihWTyV7Sra`
**Severity:** HIGH
**Files Affected:** `bot.py` (lines 867-941), `handlers/stream.py` (lines 607-681)

**Problem:**
- 62 lines of identical code duplicated across two files
- Three methods: `_add_reaction()`, `_remove_reaction()`, `_add_reactions_parallel()`
- Bug fixes must be applied in two places
- Violates DRY principle

**Solution:**
- Create `lg2slack/mixins/reactions.py` with `ReactionMixin` class
- Move all reaction logic to mixin
- Use composition in both `SlackBot` and `StreamingHandler`

**Expected Impact:**
- Eliminates 62 lines of duplication
- Single source of truth
- Easier to maintain and test

**Estimated Time:** 1-2 hours

---

## 🟡 Medium Priority

### ✅ Issue #2: Import Inside Function (stream.py)
**Status:** ✅ Complete (commit: e2c6fbe)
**Branch:** `claude/fix-import-stream-011CUomKh1oKXEihWTyV7Sra`
**Severity:** MEDIUM
**Files Affected:** `handlers/stream.py` (line 513)

**Problem:**
```python
async def _stop_slack_stream(self, ...):
    import re  # ❌ Import inside function
    text_without_images = re.sub(...)
```

**Solution:**
- Move `import re` to top of file (line ~7)

**Expected Impact:**
- Follows PEP 8 best practices
- Slightly more efficient
- Clearer dependencies

**Estimated Time:** 5 minutes

---

### ✅ Issue #3: Use Loguru for Logging (utils.py)
**Status:** ✅ Complete (commit: 593e11c)
**Branch:** `claude/switch-to-loguru-011CUomKh1oKXEihWTyV7Sra`
**Severity:** MEDIUM
**Files Affected:** `utils.py` (lines 135-136)

**Problem:**
```python
def extract_markdown_images(text: str, ...):
    import logging  # ❌ Import inside function
    logger = logging.getLogger(__name__)
```

**Solution:**
- Replace standard `logging` with `loguru`
- Use module-level logger
- Consistent with modern Python logging practices

**Expected Impact:**
- Better structured logging
- More readable logs
- Follows project preferences (loguru)

**Estimated Time:** 15 minutes

---

## 📋 Checklist

- [x] Issue #1: Extract duplicate reaction methods
- [x] Issue #2: Move import re to module level
- [x] Issue #3: Switch to loguru logging

---

## Notes

- Each fix should be on its own branch
- Run all tests after each fix (`pytest -x`)
- Update this file when status changes
- Mark items complete with ✅ when merged
