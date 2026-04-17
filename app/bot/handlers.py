from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections import OrderedDict
from contextlib import suppress
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageReactionUpdated,
)

from app.bot import (
    admin_command_flow,
    analysis_detail_flow,
    analysis_reply_flow,
    alert_command_flow,
    callback_transport_flow,
    command_menu_flow,
    command_dispatcher,
    conversation_state,
    conversation_router,
    data_account_command_flow,
    feedback_admin,
    feedback_helper_flow,
    free_text_flow,
    greeting_flow,
    group_chat_flow,
    handler_dependencies,
    followup_callback_flow,
    giveaway_command_flow,
    giveaway_menu_flow,
    llm_reply_flow,
    market_detail_flow,
    parsed_intent_executor,
    pre_route_state_executor,
    position_command_flow,
    quick_action_callback_flow,
    reply_engine,
    routed_intent_executor,
    settings_callback_flow,
    shortcut_command_flow,
    source_query_flow,
    route_text_flow,
    transport_runtime,
    trade_setup_command_flow,
)
from app.bot.keyboards import (
    alert_created_menu,
    alert_quick_menu,
    alpha_quick_menu,
    analysis_progressive_menu,
    chart_quick_menu,
    command_center_menu,
    confirm_understanding_kb,
    ema_quick_menu,
    feedback_reason_kb,
    findpair_quick_menu,
    giveaway_duration_menu,
    giveaway_menu,
    giveaway_winners_menu,
    heatmap_quick_menu,
    llm_reply_keyboard,
    news_quick_menu,
    rsi_quick_menu,
    scan_quick_menu,
    settings_menu,
    setup_quick_menu,
    simple_followup,
    smart_action_menu,
    wallet_actions,
    watch_quick_menu,
)
from app.bot.templates import (
    clarifying_question,
    correlation_template,
    cycle_template,
    giveaway_status_template,
    help_text,
    news_template,
    rsi_scan_template,
    settings_text,
    trade_plan_template,
    trade_verification_template,
    wallet_scan_template,
    watchlist_template,
)
from app.core.config import get_settings
from app.core.container import ServiceHub
from app.core.fmt import safe_html
from app.core.fred_persona import ghost as fred
from app.core.howto import try_answer_howto
from app.core.knowledge import try_answer_definition
from app.core.metrics import record_abuse, record_feedback
from app.core.nlu import (
    COMMON_WORDS_NOT_TICKERS,
    Intent,
    is_likely_english_phrase,
    parse_message,
    parse_timestamp,
)
from app.services.market_context import format_market_context

router = Router()
_settings = get_settings()
_hub: ServiceHub | None = None
_ALLOWED_OPENAI_CHAT_MODES = {"hybrid", "tool_first", "llm_first", "chat_only"}


class _LRULockCache:
    """Bounded LRU cache for per-chat asyncio locks.

    Evicts the least-recently-used *idle* lock when the cache is full,
    keeping active (locked) entries alive regardless of size.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self._cache: OrderedDict[int, asyncio.Lock] = OrderedDict()
        self._maxsize = maxsize

    def get(self, chat_id: int) -> asyncio.Lock:
        if chat_id in self._cache:
            self._cache.move_to_end(chat_id)
            return self._cache[chat_id]
        lock = asyncio.Lock()
        self._cache[chat_id] = lock
        self._cache.move_to_end(chat_id)
        # Evict the oldest idle entry when over capacity
        if len(self._cache) > self._maxsize:
            for k, v in list(self._cache.items()):
                if not v.locked():
                    del self._cache[k]
                    break
        return lock


_CHAT_LOCKS = _LRULockCache(maxsize=2000)
logger = logging.getLogger(__name__)

# Greeting fast-path ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â compiled once at import time
_GREETING_RE = re.compile(
    r"^(gm|gn|gg|gm fren|gn fren|good\s*morning|good\s*night|"
    r"hi|hey|hello|sup|yo|wassup|wagmi|lgtm|lfg|ngmi|ser|fren|anon|"
    r"wen\s*moon|wen\s*lambo|wen\s*pump|wen\s*bull|wen\s*dump|"
    r"still\s*alive|you\s*there|you\s*alive|are\s*you\s*there)[\s!?.]*$",
    re.IGNORECASE,
)
_GM_REPLIES = [
    "gm fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ charts are open, tape is moving. what are we hunting today?",
    "gm anon ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â market's breathing. drop a ticker or ask anything.",
    "gm ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ still alive, still watching. what do you need?",
    "gm fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â locked in. throw me a coin or question.",
    "gm anon. BTC still the anchor, alts still lagging dominance. what's the play?",
    "gm ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ fresh session. give me a ticker, a question, or ask what's moving.",
    "gm ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â charts loaded, alerts armed. what are we doing today?",
    "gm fren. the market doesn't care about your feelings. let's get to work.",
    "gm anon ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ new candle, new opportunity. what's on the radar?",
]
_GN_REPLIES = [
    "gn fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ set your alerts before you sleep.",
    "gn anon. the market doesn't sleep but you should.",
    "gn ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â if you haven't set alerts, do it now. i'll watch.",
    "gn fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ tape keeps printing while you rest. alerts are armed.",
]
_FEEDBACK_REASON_LABELS = {
    "thumbs_up",
    "long",
    "wrong",
    "other",
    "suggestion",
    "reaction",
}
ACTION_SYMBOL_STOPWORDS = {
    "what",
    "who",
    "hwo",
    "how",
    "are",
    "you",
    "doing",
    "coin",
    "coins",
    "overbought",
    "oversold",
    "list",
    "top",
    "news",
    "alert",
    "scan",
    "chart",
    "heatmap",
    "short",
    "long",
}
ACTION_SYMBOL_STOPWORDS.update(COMMON_WORDS_NOT_TICKERS)




def _safe_exc(exc: Exception) -> str:
    """Return exception message safe for Telegram HTML parse_mode (no raw < > & chars)."""
    return (
        str(exc)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Valid Telegram HTML tags (self-closing not needed)
_TELEGRAM_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler"}
_HTML_TAG_RE = re.compile(r"<(/?)(\w[\w\-]*)(\s[^>]*)?>", re.IGNORECASE)


def _sanitize_llm_html(text: str) -> str:
    """
    Clean LLM-generated HTML so Telegram won't reject it.
    - Strips unsupported tags (keeps their text content)
    - Closes any unclosed valid tags
    """
    if not text:
        return text

    result: list[str] = []
    open_stack: list[str] = []  # tags opened but not yet closed
    pos = 0

    for m in _HTML_TAG_RE.finditer(text):
        # Append the literal text before this tag
        result.append(text[pos:m.start()])
        pos = m.end()

        closing = m.group(1) == "/"
        tag = m.group(2).lower()
        attrs = m.group(3) or ""

        if tag not in _TELEGRAM_ALLOWED_TAGS:
            # Unsupported tag ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â drop the tag but keep nothing (content will still flow through)
            continue

        if closing:
            # Only emit the closing tag if we actually opened this tag
            if tag in open_stack:
                # Close any tags opened after this one (auto-close nested unclosed tags)
                while open_stack and open_stack[-1] != tag:
                    result.append(f"</{open_stack.pop()}>")
                if open_stack:
                    open_stack.pop()
                result.append(f"</{tag}>")
        else:
            result.append(f"<{tag}{attrs}>")
            # <a> and block-level tags need tracking; void-like usage is rare in LLM output
            open_stack.append(tag)

    # Append remaining text
    result.append(text[pos:])

    # Close any still-open tags in reverse order
    for tag in reversed(open_stack):
        result.append(f"</{tag}>")

    return "".join(result)


def _build_communication_memory_block(settings: dict | None) -> str:
    return llm_reply_flow.build_communication_memory_block(settings)


async def _append_last_symbol(chat_id: int, symbol: str) -> None:
    """Append symbol to user's last_symbols (max 5) for memory."""
    hub = _require_hub()
    settings = await hub.user_service.get_settings(chat_id)
    last = list(settings.get("last_symbols") or [])
    if not isinstance(last, list):
        last = []
    sym = (symbol or "").upper().strip()
    if not sym:
        return
    last = [sym] + [x for x in last if str(x).upper() != sym][:4]
    await hub.user_service.update_settings(chat_id, {"last_symbols": last})


async def _send_llm_reply(
    message: Message,
    reply: str,
    settings: dict | None = None,
    user_message: str | None = None,
    add_quick_replies: bool = True,
    analytics: dict | None = None,
) -> None:
    await reply_engine.send_llm_reply(
        message=message,
        reply=reply,
        hub=_require_hub(),
        sanitize_html=_sanitize_llm_html,
        llm_reply_keyboard_factory=llm_reply_keyboard,
        confirm_understanding_kb_factory=confirm_understanding_kb,
        chat_mode=_openai_chat_mode(),
        settings=settings,
        user_message=user_message,
        add_quick_replies=add_quick_replies,
        analytics=analytics,
    )


def _transport_runtime_dependencies() -> transport_runtime.TransportRuntimeDependencies:
    hub = _require_hub()
    return transport_runtime.TransportRuntimeDependencies(
        cache=hub.cache,
        rate_limiter=hub.rate_limiter,
        logger=logger,
        request_rate_limit_per_minute=int(_settings.request_rate_limit_per_minute),
        abuse_strike_window_sec=int(_settings.abuse_strike_window_sec),
        abuse_strikes_to_block=int(_settings.abuse_strikes_to_block),
        abuse_block_ttl_sec=int(_settings.abuse_block_ttl_sec),
        record_abuse=record_abuse,
        chat_lock=_CHAT_LOCKS.get,
    )


async def _acquire_callback_once(callback: CallbackQuery, ttl: int = 60 * 30) -> bool:
    return await transport_runtime.acquire_callback_once(
        callback,
        deps=_transport_runtime_dependencies(),
        ttl=ttl,
    )


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    await transport_runtime.run_with_typing_lock(
        bot,
        chat_id,
        runner,
        deps=_transport_runtime_dependencies(),
    )


def init_handlers(hub: ServiceHub) -> None:
    global _hub
    _hub = hub


def _parse_int_list(value, fallback: list[int]) -> list[int]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out or fallback


def _source_query_dependencies() -> source_query_flow.SourceQueryFlowDependencies:
    return source_query_flow.SourceQueryFlowDependencies(cache=_require_hub().cache)


def _extract_action_symbol_hint(text: str) -> str | None:
    if is_likely_english_phrase(text):
        return None
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in ACTION_SYMBOL_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


def _parse_tf_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out = [x for x in items if x]
    return out or fallback


def _analysis_timeframes_from_settings(settings: dict) -> list[str]:
    if _settings.analysis_fast_mode:
        return _parse_tf_list(settings.get("preferred_timeframe", "1h"), ["1h"])
    return _parse_tf_list(settings.get("preferred_timeframes", settings.get("preferred_timeframe", "1h,4h")), ["1h", "4h"])


def _require_hub() -> ServiceHub:
    if _hub is None:
        raise RuntimeError("Handlers not initialized")
    return _hub


def _openai_chat_mode() -> str:
    mode = str(_settings.openai_chat_mode or "hybrid").strip().lower()
    if mode not in _ALLOWED_OPENAI_CHAT_MODES:
        return "hybrid"
    return mode


def _group_chat_flow_dependencies() -> group_chat_flow.GroupChatFlowDependencies:
    hub = _require_hub()
    return group_chat_flow.GroupChatFlowDependencies(
        cache=hub.cache,
        bot=hub.bot,
        admin_ids_list=_settings.admin_ids_list,
        parse_message=parse_message,
        clear_intent_excluded={Intent.UNKNOWN, Intent.SMALLTALK, Intent.HELP, Intent.START},
    )


def _greeting_flow_dependencies() -> greeting_flow.GreetingFlowDependencies:
    return greeting_flow.GreetingFlowDependencies(
        analysis_service=_require_hub().analysis_service,
        choose_reply=random.choice,
        gm_replies=_GM_REPLIES,
        weekend_warning_text="<i>weekend session - liquidity is thin, spreads are wide. size accordingly.</i>",
        now_utc=lambda: datetime.now(UTC),
    )


def _dependency_factory() -> handler_dependencies.HandlerDependencyFactory:
    return handler_dependencies.HandlerDependencyFactory(
        hub=_require_hub(),
        settings_obj=_settings,
        logger=logger,
        feedback_reason_labels=_FEEDBACK_REASON_LABELS,
        transport_runtime_deps=_transport_runtime_dependencies(),
        source_query_deps=_source_query_dependencies(),
        group_chat_deps=_group_chat_flow_dependencies(),
        greeting_deps=_greeting_flow_dependencies(),
        helpers=SimpleNamespace(
            acquire_callback_once=_acquire_callback_once,
            alert_created_menu=alert_created_menu,
            alert_quick_menu=alert_quick_menu,
            alpha_quick_menu=alpha_quick_menu,
            analysis_progressive_menu=analysis_progressive_menu,
            analysis_symbol_followup_kb=_analysis_symbol_followup_kb,
            analysis_timeframes_from_settings=_analysis_timeframes_from_settings,
            append_last_symbol=_append_last_symbol,
            as_float=_as_float,
            as_float_list=_as_float_list,
            as_int=_as_int,
            blocked_message=_blocked_message_text,
            blocked_rate_limit_message=_blocked_rate_limit_message_text,
            bot_meta_re=_BOT_META_RE,
            buffered_input_file_cls=BufferedInputFile,
            build_communication_memory_block=_build_communication_memory_block,
            busy_notice="still on it fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â give me a few seconds.",
            chart_quick_menu=chart_quick_menu,
            choose_reply=random.choice,
            clarifying_question=clarifying_question,
            clear_cmd_wizard=_cmd_wizard_clear,
            clear_pending_alert=_clear_pending_alert,
            clear_wizard=_wizard_clear,
            cycle_template=cycle_template,
            define_keyboard=_define_keyboard,
            dispatch_command_text=_dispatch_command_text,
            ema_quick_menu=ema_quick_menu,
            extract_action_symbol_hint=_extract_action_symbol_hint,
            extract_symbol=_extract_symbol,
            feedback_reason_kb=feedback_reason_kb,
            findpair_quick_menu=findpair_quick_menu,
            format_as_ghost=fred.format_as_ghost,
            format_feedback_summary=_format_feedback_summary,
            format_market_context=format_market_context,
            format_quality_summary=_format_quality_summary,
            format_reply_stats_summary=_format_reply_stats_summary,
            get_chat_history=_get_chat_history,
            get_cmd_wizard=_cmd_wizard_get,
            get_pending_alert=_get_pending_alert,
            get_wizard=_wizard_get,
            ghost_persona=fred,
            giveaway_duration_menu=giveaway_duration_menu,
            giveaway_menu=giveaway_menu,
            giveaway_status_template=giveaway_status_template,
            giveaway_winners_menu=giveaway_winners_menu,
            gn_replies=_GN_REPLIES,
            greeting_re=_GREETING_RE,
            group_free_talk_enabled=group_chat_flow.group_free_talk_enabled,
            group_is_admin=group_chat_flow.is_group_admin,
            handle_free_text_flow=_handle_free_text_flow,
            handle_parsed_intent=_handle_parsed_intent,
            handle_pre_route_state=_handle_pre_route_state,
            handle_routed_intent=_handle_routed_intent,
            heatmap_quick_menu=heatmap_quick_menu,
            is_bot_admin=_is_bot_admin,
            is_definition_question=_is_definition_question,
            is_likely_english_phrase=is_likely_english_phrase,
            is_reply_to_bot=group_chat_flow.is_reply_to_bot,
            llm_analysis_reply=_llm_analysis_reply,
            llm_fallback_reply=_llm_fallback_reply,
            llm_followup_reply=_llm_followup_reply,
            llm_market_chat_reply=_llm_market_chat_reply,
            llm_reply_keyboard=llm_reply_keyboard,
            llm_route_message=_llm_route_message,
            looks_like_analysis_followup=_looks_like_analysis_followup,
            looks_like_clear_intent=group_chat_flow.looks_like_clear_intent,
            mentions_bot=group_chat_flow.mentions_bot,
            news_quick_menu=news_quick_menu,
            news_template=news_template,
            normalize_symbol_value=_normalize_symbol_value,
            now_utc=lambda: datetime.now(UTC),
            openai_chat_mode=_openai_chat_mode,
            parse_duration_to_seconds=_parse_duration_to_seconds,
            parse_int_list=_parse_int_list,
            parse_message=parse_message,
            parse_timestamp=parse_timestamp,
            pause=asyncio.sleep,
            plain_text_prompt="Send a request in plain text, e.g. `SOL long`, `cpi news`, `chart btc 1h`, or `alert me when SOL hits 50`.",
            rate_limit_notice="slow down fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â rate limit hit. resets in ~1 min.",
            recent_analysis_context=_recent_analysis_context,
            record_abuse=record_abuse,
            record_feedback_metric=record_feedback,
            remember_analysis_context=_remember_analysis_context,
            render_analysis_text=_render_analysis_text,
            route_free_text=conversation_router.handle_free_text_routing,
            rsi_quick_menu=rsi_quick_menu,
            rsi_scan_template=rsi_scan_template,
            run_with_typing_lock=_run_with_typing_lock,
            safe_exc=_safe_exc,
            safe_html=safe_html,
            sanitize_html=_sanitize_llm_html,
            save_trade_check=_save_trade_check,
            scan_quick_menu=scan_quick_menu,
            send_ghost_analysis=_send_ghost_analysis,
            send_llm_reply=_send_llm_reply,
            set_cmd_wizard=_cmd_wizard_set,
            set_group_free_talk=group_chat_flow.set_group_free_talk,
            set_pending_alert=_set_pending_alert,
            set_wizard=_wizard_set,
            settings_menu=settings_menu,
            settings_text=settings_text,
            setup_quick_menu=setup_quick_menu,
            simple_followup=simple_followup,
            smart_action_menu=smart_action_menu,
            strip_bot_mention=group_chat_flow.strip_bot_mention,
            trade_math_payload=_trade_math_payload,
            trade_plan_template=trade_plan_template,
            trade_verification_template=trade_verification_template,
            try_answer_definition=try_answer_definition,
            try_answer_howto=try_answer_howto,
            wallet_actions=wallet_actions,
            wallet_scan_template=wallet_scan_template,
            watch_quick_menu=watch_quick_menu,
            watchlist_template=watchlist_template,
        ),
    )


async def _run_callback_handler(callback: CallbackQuery, runner) -> None:
    await callback_transport_flow.run_deduped_callback(
        callback=callback,
        acquire_callback_once=_acquire_callback_once,
        runner=runner,
    )


def _blocked_message_text(ttl: int) -> str:
    ttl_txt = ""
    if ttl and ttl > 0:
        ttl_txt = f" (try again in ~{max(ttl // 60, 1)} min)"
    return f"You're temporarily blocked for spam.{ttl_txt}"


def _blocked_rate_limit_message_text(ttl: int) -> str:
    ttl_txt = ""
    if ttl and ttl > 0:
        ttl_txt = f" (try again in ~{max(ttl // 60, 1)} min)"
    return f"rate limit hit. you're blocked for a bit{ttl_txt}."


def _route_text_flow_dependencies() -> route_text_flow.RouteTextFlowDependencies:
    hub = _require_hub()
    source_deps = _source_query_dependencies()
    group_deps = _group_chat_flow_dependencies()
    greeting_deps = _greeting_flow_dependencies()
    return route_text_flow.RouteTextFlowDependencies(
        hub=hub,
        logger=logger,
        record_abuse=record_abuse,
        is_blocked_subject=partial(transport_runtime.is_blocked_subject, deps=_transport_runtime_dependencies()),
        blocked_notice_ttl=partial(transport_runtime.blocked_notice_ttl, deps=_transport_runtime_dependencies()),
        is_group_admin=partial(group_chat_flow.is_group_admin, deps=group_deps),
        set_group_free_talk=partial(group_chat_flow.set_group_free_talk, deps=group_deps),
        group_free_talk_enabled=partial(group_chat_flow.group_free_talk_enabled, deps=group_deps),
        mentions_bot=group_chat_flow.mentions_bot,
        strip_bot_mention=group_chat_flow.strip_bot_mention,
        is_reply_to_bot=partial(group_chat_flow.is_reply_to_bot, bot_id=hub.bot.id),
        looks_like_clear_intent=partial(group_chat_flow.looks_like_clear_intent, deps=group_deps),
        is_source_query=source_query_flow.is_source_query,
        source_reply_for_chat=partial(source_query_flow.source_reply_for_chat, deps=source_deps),
        acquire_message_once=partial(transport_runtime.acquire_message_once, deps=_transport_runtime_dependencies()),
        check_request_limit=partial(transport_runtime.check_request_limit, deps=_transport_runtime_dependencies()),
        record_strike_and_maybe_block=partial(transport_runtime.record_strike_and_maybe_block, deps=_transport_runtime_dependencies()),
        greeting_re=_GREETING_RE,
        choose_reply=random.choice,
        gn_replies=_GN_REPLIES,
        market_aware_gm_reply=partial(greeting_flow.market_aware_gm_reply, deps=greeting_deps),
        chat_lock=_CHAT_LOCKS.get,
        typing_loop=transport_runtime.typing_loop,
        handle_pre_route_state=_handle_pre_route_state,
        handle_free_text_flow=_handle_free_text_flow,
        safe_exc=_safe_exc,
        now_utc=lambda: datetime.now(UTC),
        blocked_message=_blocked_message_text,
        blocked_rate_limit_message=_blocked_rate_limit_message_text,
        rate_limit_notice="slow down fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â rate limit hit. resets in ~1 min.",
        plain_text_prompt="Send a request in plain text, e.g. `SOL long`, `cpi news`, `chart btc 1h`, or `alert me when SOL hits 50`.",
        busy_notice="still on it fren ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â give me a few seconds.",
    )


async def _get_chat_history(chat_id: int) -> list[dict[str, str]]:
    return await llm_reply_flow.get_chat_history(
        cache=_require_hub().cache,
        chat_id=chat_id,
    )


async def _append_chat_history(chat_id: int, role: str, content: str) -> None:
    await llm_reply_flow.append_chat_history(
        cache=_require_hub().cache,
        chat_id=chat_id,
        role=role,
        content=content,
        turns=max(int(_settings.openai_chat_history_turns), 1),
    )


def _llm_reply_flow_dependencies() -> llm_reply_flow.LlmReplyFlowDependencies:
    return llm_reply_flow.LlmReplyFlowDependencies(
        hub=_require_hub(),
        openai_chat_history_turns=int(_settings.openai_chat_history_turns),
        openai_max_output_tokens=int(_settings.openai_max_output_tokens),
        openai_temperature=float(_settings.openai_temperature),
        bot_meta_re=_BOT_META_RE,
        try_answer_definition=try_answer_definition,
        try_answer_howto=try_answer_howto,
        format_market_context=format_market_context,
    )


def _analysis_reply_flow_dependencies() -> analysis_reply_flow.AnalysisReplyFlowDependencies:
    return analysis_reply_flow.AnalysisReplyFlowDependencies(
        format_as_ghost=fred.format_as_ghost,
        llm_analysis_reply=_llm_analysis_reply,
        trade_plan_template=trade_plan_template,
        analysis_progressive_menu=analysis_progressive_menu,
        pause=asyncio.sleep,
    )


async def _remember_analysis_context(chat_id: int, symbol: str, direction: str | None, payload: dict) -> None:
    await analysis_reply_flow.remember_analysis_context(
        cache=_require_hub().cache,
        chat_id=chat_id,
        symbol=symbol,
        direction=direction,
        payload=payload,
    )


async def _recent_analysis_context(chat_id: int) -> dict | None:
    return await analysis_reply_flow.recent_analysis_context(
        cache=_require_hub().cache,
        chat_id=chat_id,
    )


def _looks_like_analysis_followup(text: str, context: dict | None) -> bool:
    return analysis_reply_flow.looks_like_analysis_followup(text, context)


async def _llm_analysis_reply(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    chat_id: int | None,
) -> str | None:
    return await llm_reply_flow.llm_analysis_reply(
        payload=payload,
        symbol=symbol,
        direction=direction,
        chat_id=chat_id,
        deps=_llm_reply_flow_dependencies(),
    )


async def _llm_followup_reply(
    user_text: str,
    context: dict,
    *,
    chat_id: int,
) -> str | None:
    return await llm_reply_flow.llm_followup_reply(
        user_text,
        context,
        chat_id=chat_id,
        deps=_llm_reply_flow_dependencies(),
    )


async def _render_analysis_text(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    settings: dict,
    chat_id: int,
    detailed: bool = False,
) -> str:
    return await analysis_reply_flow.render_analysis_text(
        payload=payload,
        symbol=symbol,
        direction=direction,
        settings=settings,
        chat_id=chat_id,
        detailed=detailed,
        deps=_analysis_reply_flow_dependencies(),
    )


async def _send_ghost_analysis(message: Message, symbol: str, text: str, direction: str | None = None) -> None:
    await analysis_reply_flow.send_ghost_analysis(
        message,
        symbol,
        text,
        direction=direction,
        deps=_analysis_reply_flow_dependencies(),
    )


def _define_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Analyze 1h", callback_data="define:analyze:1h"),
                InlineKeyboardButton(text="Analyze 4h", callback_data="define:analyze:4h"),
            ],
            [
                InlineKeyboardButton(text="Chart 1h", callback_data="define:chart:1h"),
                InlineKeyboardButton(text="Heatmap", callback_data="define:heatmap"),
            ],
            [
                InlineKeyboardButton(text="Set Alert", callback_data="define:alert"),
                InlineKeyboardButton(text="Top Overbought 1h", callback_data="top:overbought:1h"),
            ],
            [
                InlineKeyboardButton(text="Top Oversold 1h", callback_data="top:oversold:1h"),
                InlineKeyboardButton(text="Latest News", callback_data="define:news"),
            ],
        ]
    )


async def _llm_fallback_reply(user_text: str, settings: dict | None = None, chat_id: int | None = None) -> str | None:
    return await llm_reply_flow.llm_fallback_reply(
        user_text,
        settings=settings,
        chat_id=chat_id,
        deps=_llm_reply_flow_dependencies(),
    )



def _is_definition_question(text: str) -> bool:
    return llm_reply_flow.is_definition_question(text)


async def _llm_market_chat_reply(
    user_text: str,
    settings: dict | None = None,
    chat_id: int | None = None,
) -> str | None:
    return await llm_reply_flow.llm_market_chat_reply(
        user_text,
        settings=settings,
        chat_id=chat_id,
        deps=_llm_reply_flow_dependencies(),
    )


def _parse_duration_to_seconds(raw: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", raw.lower())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return value * mult


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _as_float_list(value) -> list[float]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.findall(r"[0-9]+(?:\.[0-9]+)?", value)
    else:
        raw = []
    out: list[float] = []
    for item in raw:
        v = _as_float(item)
        if v is not None:
            out.append(float(v))
    return out


def _infer_direction(entry: float, targets: list[float], explicit: str | None) -> str:
    side = (explicit or "").strip().lower()
    if side in {"long", "short"}:
        return side
    if not targets:
        return "long"
    return "long" if float(targets[0]) >= float(entry) else "short"


def _trade_math_payload(
    *,
    entry: float,
    stop: float,
    targets: list[float],
    direction: str | None,
    margin_usd: float | None,
    leverage: float | None,
    symbol: str | None = None,
) -> dict:
    e = float(entry)
    s = float(stop)
    tps = [float(x) for x in targets if float(x) > 0]
    if not tps:
        raise RuntimeError("Need at least one target.")
    risk = abs(e - s)
    if risk <= 0:
        raise RuntimeError("Entry and stop cannot be the same.")
    side = _infer_direction(e, tps, direction)

    rows: list[dict] = []
    for tp in tps:
        reward = (tp - e) if side == "long" else (e - tp)
        r_mult = reward / risk
        rows.append({"tp": round(tp, 8), "r_multiple": round(r_mult, 3)})
    best_r = max(row["r_multiple"] for row in rows)

    payload: dict = {
        "symbol": symbol or "",
        "direction": side,
        "entry": round(e, 8),
        "stop": round(s, 8),
        "targets": [round(x, 8) for x in tps],
        "risk_per_unit": round(risk, 8),
        "rows": rows,
        "best_r": round(best_r, 3),
    }

    if margin_usd and leverage and margin_usd > 0 and leverage > 0:
        notional = float(margin_usd) * float(leverage)
        qty = notional / e

        def _pnl(exit_price: float) -> float:
            if side == "long":
                return (exit_price - e) * qty
            return (e - exit_price) * qty

        payload["position"] = {
            "margin_usd": round(float(margin_usd), 2),
            "leverage": round(float(leverage), 2),
            "notional_usd": round(notional, 2),
            "qty": round(qty, 8),
            "stop_pnl_usd": round(_pnl(s), 2),
            "tp_pnls": [{"tp": round(tp, 8), "pnl_usd": round(_pnl(tp), 2)} for tp in tps],
        }

    return payload


def _extract_symbol(params: dict) -> str | None:
    for key in ("symbol", "asset", "ticker", "coin"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper().lstrip("$")
    return None


def _normalize_symbol_value(val) -> str | None:
    """Best-effort normalization for symbol-like values coming from multiple sources."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s.upper().lstrip("$") if s else None
    # NLU may pass a Symbol-like object with `.base`
    base = getattr(val, "base", None)
    if isinstance(base, str) and base.strip():
        return base.strip().upper().lstrip("$")
    return None


async def _llm_route_message(user_text: str) -> dict | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None
    try:
        payload = await hub.llm_client.route_message(user_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _analysis_symbol_followup_kb() -> InlineKeyboardMarkup:
    return simple_followup(
        [
            ("BTC", "quick:analysis:BTC"),
            ("ETH", "quick:analysis:ETH"),
            ("SOL", "quick:analysis:SOL"),
        ]
    )


async def _dispatch_command_text(message: Message, synthetic_text: str) -> bool:
    return await command_dispatcher.dispatch_command_text(
        message=message,
        synthetic_text=synthetic_text,
        hub=_require_hub(),
        handle_parsed_intent=_handle_parsed_intent,
        llm_fallback_reply=_llm_fallback_reply,
        send_llm_reply=_send_llm_reply,
        clarifying_question=clarifying_question,
        extract_action_symbol_hint=_extract_action_symbol_hint,
        smart_action_menu=smart_action_menu,
        analysis_symbol_followup_kb=_analysis_symbol_followup_kb,
        safe_exc=_safe_exc,
        alert_created_menu=alert_created_menu,
    )


def _routed_intent_dependencies() -> routed_intent_executor.RoutedIntentDependencies:
    source_deps = _source_query_dependencies()
    return routed_intent_executor.RoutedIntentDependencies(
        hub=_require_hub(),
        openai_router_min_confidence=_settings.openai_router_min_confidence,
        bot_meta_re=_BOT_META_RE,
        try_answer_howto=try_answer_howto,
        llm_fallback_reply=_llm_fallback_reply,
        llm_market_chat_reply=_llm_market_chat_reply,
        send_llm_reply=_send_llm_reply,
        as_int=_as_int,
        as_float=_as_float,
        as_float_list=_as_float_list,
        extract_symbol=_extract_symbol,
        normalize_symbol_value=_normalize_symbol_value,
        analysis_timeframes_from_settings=_analysis_timeframes_from_settings,
        parse_int_list=_parse_int_list,
        append_last_symbol=_append_last_symbol,
        remember_analysis_context=_remember_analysis_context,
        remember_source_context=partial(source_query_flow.remember_source_context, deps=source_deps),
        render_analysis_text=_render_analysis_text,
        send_ghost_analysis=_send_ghost_analysis,
        safe_exc=_safe_exc,
        parse_duration_to_seconds=_parse_duration_to_seconds,
        trade_math_payload=_trade_math_payload,
        feature_flags_set=_settings.feature_flags_set,
    )


async def _handle_routed_intent(message: Message, settings: dict, route: dict) -> bool:
    return await routed_intent_executor.execute_routed_intent(
        message=message,
        settings=settings,
        route=route,
        deps=_routed_intent_dependencies(),
    )


def _pre_route_state_dependencies() -> pre_route_state_executor.PreRouteStateDependencies:
    helper_deps = _feedback_helper_dependencies()
    source_deps = _source_query_dependencies()
    return pre_route_state_executor.PreRouteStateDependencies(
        hub=_require_hub(),
        as_int=_as_int,
        dispatch_command_text=_dispatch_command_text,
        get_pending_feedback_suggestion=lambda chat_id: feedback_helper_flow.get_pending_feedback_suggestion(
            chat_id=chat_id,
            deps=helper_deps,
        ),
        clear_pending_feedback_suggestion=lambda chat_id: feedback_helper_flow.clear_pending_feedback_suggestion(
            chat_id=chat_id,
            deps=helper_deps,
        ),
        log_feedback_event=lambda **kwargs: feedback_helper_flow.log_feedback_event(
            deps=helper_deps,
            **kwargs,
        ),
        notify_admins_negative_feedback=lambda **kwargs: feedback_helper_flow.notify_admins_negative_feedback(
            deps=helper_deps,
            **kwargs,
        ),
        get_cmd_wizard=_cmd_wizard_get,
        clear_cmd_wizard=_cmd_wizard_clear,
        get_wizard=_wizard_get,
        set_wizard=_wizard_set,
        clear_wizard=_wizard_clear,
        parse_timestamp=parse_timestamp,
        save_trade_check=_save_trade_check,
        remember_source_context=partial(source_query_flow.remember_source_context, deps=source_deps),
        get_pending_alert=_get_pending_alert,
        clear_pending_alert=_clear_pending_alert,
        trade_verification_template=trade_verification_template,
        giveaway_menu=giveaway_menu,
        alert_created_menu=alert_created_menu,
    )


async def _handle_pre_route_state(message: Message, text: str, chat_id: int) -> bool:
    return await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text=text,
        chat_id=chat_id,
        deps=_pre_route_state_dependencies(),
    )


def _free_text_flow_dependencies() -> free_text_flow.FreeTextFlowDependencies:
    return free_text_flow.FreeTextFlowDependencies(
        hub=_require_hub(),
        parse_message=parse_message,
        openai_chat_mode=_openai_chat_mode,
        route_free_text=conversation_router.handle_free_text_routing,
        send_llm_reply=_send_llm_reply,
        get_chat_history=_get_chat_history,
        dispatch_command_text=_dispatch_command_text,
        recent_analysis_context=_recent_analysis_context,
        looks_like_analysis_followup=_looks_like_analysis_followup,
        llm_followup_reply=_llm_followup_reply,
        llm_market_chat_reply=_llm_market_chat_reply,
        llm_route_message=_llm_route_message,
        handle_routed_intent=_handle_routed_intent,
        handle_parsed_intent=_handle_parsed_intent,
        is_definition_question=_is_definition_question,
        is_likely_english_phrase=is_likely_english_phrase,
        extract_action_symbol_hint=_extract_action_symbol_hint,
        clarifying_question=clarifying_question,
        smart_action_menu=smart_action_menu,
        analysis_symbol_followup_kb=_analysis_symbol_followup_kb,
        define_keyboard=_define_keyboard,
        pause=asyncio.sleep,
    )


async def _handle_free_text_flow(message: Message, text: str, chat_id: int, start_ts: datetime) -> bool:
    return await free_text_flow.handle_free_text_flow(
        message=message,
        text=text,
        chat_id=chat_id,
        start_ts=start_ts,
        deps=_free_text_flow_dependencies(),
    )


async def _get_pending_alert(chat_id: int) -> str | None:
    return await conversation_state.get_pending_alert(_require_hub().cache, chat_id)


async def _set_pending_alert(chat_id: int, symbol: str) -> None:
    await conversation_state.set_pending_alert(_require_hub().cache, chat_id, symbol)


async def _clear_pending_alert(chat_id: int) -> None:
    await conversation_state.clear_pending_alert(_require_hub().cache, chat_id)


async def _wizard_get(chat_id: int) -> dict | None:
    return await conversation_state.get_tradecheck_wizard(_require_hub().cache, chat_id)


async def _wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    await conversation_state.set_tradecheck_wizard(_require_hub().cache, chat_id, payload, ttl=ttl)


async def _wizard_clear(chat_id: int) -> None:
    await conversation_state.clear_tradecheck_wizard(_require_hub().cache, chat_id)


async def _cmd_wizard_get(chat_id: int) -> dict | None:
    return await conversation_state.get_command_wizard(_require_hub().cache, chat_id)


async def _cmd_wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    await conversation_state.set_command_wizard(_require_hub().cache, chat_id, payload, ttl=ttl)


async def _cmd_wizard_clear(chat_id: int) -> None:
    await conversation_state.clear_command_wizard(_require_hub().cache, chat_id)


async def _save_trade_check(chat_id: int, data: dict, result: dict) -> None:
    hub = _require_hub()
    await conversation_state.save_trade_check(
        ensure_user=hub.user_service.ensure_user,
        chat_id=chat_id,
        data=data,
        result=result,
    )


def _command_menu_flow_dependencies() -> command_menu_flow.CommandMenuFlowDependencies:
    return command_menu_flow.CommandMenuFlowDependencies(
        hub=_require_hub(),
        dispatch_command_text=_dispatch_command_text,
        run_with_typing_lock=_run_with_typing_lock,
        set_cmd_wizard=_cmd_wizard_set,
        set_wizard=_wizard_set,
        as_int=_as_int,
        alpha_quick_menu=alpha_quick_menu,
        watch_quick_menu=watch_quick_menu,
        chart_quick_menu=chart_quick_menu,
        heatmap_quick_menu=heatmap_quick_menu,
        rsi_quick_menu=rsi_quick_menu,
        ema_quick_menu=ema_quick_menu,
        news_quick_menu=news_quick_menu,
        alert_quick_menu=alert_quick_menu,
        findpair_quick_menu=findpair_quick_menu,
        setup_quick_menu=setup_quick_menu,
        scan_quick_menu=scan_quick_menu,
        giveaway_menu=giveaway_menu,
        rsi_scan_template=rsi_scan_template,
        logger=logger,
    )


def _giveaway_menu_flow_dependencies() -> giveaway_menu_flow.GiveawayMenuFlowDependencies:
    return giveaway_menu_flow.GiveawayMenuFlowDependencies(
        hub=_require_hub(),
        run_with_typing_lock=_run_with_typing_lock,
        set_cmd_wizard=_cmd_wizard_set,
        as_int=_as_int,
        giveaway_duration_menu=giveaway_duration_menu,
        giveaway_winners_menu=giveaway_winners_menu,
        giveaway_status_template=giveaway_status_template,
    )


def _followup_callback_dependencies() -> followup_callback_flow.FollowupCallbackDependencies:
    return followup_callback_flow.FollowupCallbackDependencies(
        hub=_require_hub(),
        sanitize_html=_sanitize_llm_html,
        llm_reply_keyboard=llm_reply_keyboard,
    )


def _confirm_clear_alerts_dependencies() -> followup_callback_flow.ConfirmClearAlertsDependencies:
    return followup_callback_flow.ConfirmClearAlertsDependencies(
        clear_user_alerts=_require_hub().alerts_service.clear_user_alerts,
    )


def _settings_callback_dependencies() -> settings_callback_flow.SettingsCallbackDependencies:
    hub = _require_hub()
    return settings_callback_flow.SettingsCallbackDependencies(
        get_user_settings=hub.user_service.get_settings,
        update_user_settings=hub.user_service.update_settings,
        settings_text=settings_text,
        settings_menu=settings_menu,
    )


def _analysis_detail_flow_dependencies() -> analysis_detail_flow.AnalysisDetailFlowDependencies:
    source_deps = _source_query_dependencies()
    return analysis_detail_flow.AnalysisDetailFlowDependencies(
        hub=_require_hub(),
        set_pending_alert=_set_pending_alert,
        run_with_typing_lock=_run_with_typing_lock,
        analysis_timeframes_from_settings=_analysis_timeframes_from_settings,
        parse_int_list=_parse_int_list,
        remember_analysis_context=_remember_analysis_context,
        remember_source_context=partial(source_query_flow.remember_source_context, deps=source_deps),
        render_analysis_text=_render_analysis_text,
        send_ghost_analysis=_send_ghost_analysis,
    )


def _market_detail_flow_dependencies() -> market_detail_flow.MarketDetailFlowDependencies:
    source_deps = _source_query_dependencies()
    return market_detail_flow.MarketDetailFlowDependencies(
        hub=_require_hub(),
        feature_flags_set=_settings.feature_flags_set,
        run_with_typing_lock=_run_with_typing_lock,
        remember_source_context=partial(source_query_flow.remember_source_context, deps=source_deps),
    )


def _quick_action_callback_dependencies() -> quick_action_callback_flow.QuickActionCallbackDependencies:
    source_deps = _source_query_dependencies()
    return quick_action_callback_flow.QuickActionCallbackDependencies(
        hub=_require_hub(),
        run_with_typing_lock=_run_with_typing_lock,
        analysis_timeframes_from_settings=_analysis_timeframes_from_settings,
        parse_int_list=_parse_int_list,
        remember_analysis_context=_remember_analysis_context,
        remember_source_context=partial(source_query_flow.remember_source_context, deps=source_deps),
        render_analysis_text=_render_analysis_text,
        send_ghost_analysis=_send_ghost_analysis,
        set_pending_alert=_set_pending_alert,
        as_int=_as_int,
        rsi_scan_template=rsi_scan_template,
        news_template=news_template,
    )


def _admin_command_flow_dependencies() -> admin_command_flow.AdminCommandFlowDependencies:
    return _dependency_factory().admin_command_flow()


def _shortcut_command_flow_dependencies() -> shortcut_command_flow.ShortcutCommandFlowDependencies:
    return _dependency_factory().shortcut_command_flow()


def _alert_command_flow_dependencies() -> alert_command_flow.AlertCommandFlowDependencies:
    return _dependency_factory().alert_command_flow()


def _position_command_flow_dependencies() -> position_command_flow.PositionCommandFlowDependencies:
    return _dependency_factory().position_command_flow()


def _data_account_command_flow_dependencies() -> data_account_command_flow.DataAccountCommandFlowDependencies:
    return _dependency_factory().data_account_command_flow()


def _trade_setup_command_flow_dependencies() -> trade_setup_command_flow.TradeSetupCommandFlowDependencies:
    return _dependency_factory().trade_setup_command_flow()


def _giveaway_command_flow_dependencies() -> giveaway_command_flow.GiveawayCommandFlowDependencies:
    return _dependency_factory().giveaway_command_flow()


def _feedback_helper_dependencies() -> feedback_helper_flow.FeedbackHelperDependencies:
    return _dependency_factory().feedback_helper()


def _parsed_intent_dependencies() -> parsed_intent_executor.ParsedIntentDependencies:
    return _dependency_factory().parsed_intent()


async def _handle_parsed_intent(message: Message, parsed, settings: dict) -> bool:
    return await parsed_intent_executor.execute_parsed_intent(
        message=message,
        parsed=parsed,
        settings=settings,
        deps=_parsed_intent_dependencies(),
    )


@router.message(Command("start"))
async def start_cmd(message: Message) -> None:
    hub = _require_hub()
    user = await hub.user_service.ensure_user(message.chat.id)
    name = message.from_user.first_name if message.from_user else "fren"
    chat_id = message.chat.id

    # Detect new user: created_at within the last 15 seconds
    try:
        is_new = (datetime.utcnow() - user.created_at).total_seconds() < 15
    except Exception:
        is_new = False

    if is_new:
        await message.answer(
            f"gm <b>{name}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹\n\n"
            "i'm <b>ghost</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â your on-chain trading assistant. i live in the market 24/7 so you don't have to.\n\n"
            "try something like:\n"
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· <code>BTC 4h</code> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â full technical analysis\n"
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· <code>ping me when ETH hits 2000</code> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â price alert\n"
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· <code>coins to watch</code> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â top movers watchlist\n"
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· <code>why is BTC pumping</code> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â live market read\n\n"
            "<i>short questions get short answers. send a ticker for a deep dive. tap a button to start.</i>",
            reply_markup=smart_action_menu(),
        )
    else:
        # Returning user ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â check for session continuity
        continuity = ""
        with suppress(Exception):
            last_ctx = await hub.cache.get_json(f"last_analysis_context:{chat_id}")
            if isinstance(last_ctx, dict) and last_ctx.get("symbol"):
                sym = str(last_ctx["symbol"]).upper()
                dir_txt = last_ctx.get("direction") or ""
                direction_part = f" {dir_txt}" if dir_txt else ""
                continuity = f"\n\n<i>last time you were watching <b>{sym}{direction_part}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â want a fresh read?</i>"
        await message.answer(
            f"wb back {name}. ghost is live.{continuity}\n\n"
            "drop a ticker, ask a question, or tap a button.",
            reply_markup=smart_action_menu(),
        )


@router.message(Command("help"))
async def help_cmd(message: Message) -> None:
    await message.answer(help_text(), reply_markup=command_center_menu())


@router.message(Command("admins"))
async def admins_cmd(message: Message) -> None:
    await admin_command_flow.handle_admins_command(message=message, deps=_admin_command_flow_dependencies())


def _is_bot_admin(message: Message) -> bool:
    if not message.from_user:
        return False
    return int(message.from_user.id) in set(_settings.admin_ids_list())


def _format_feedback_summary(summary: dict) -> str:
    hours = int(summary.get("hours") or 24)
    total = int(summary.get("total") or 0)
    sentiments = dict(summary.get("sentiments") or {})
    positives = int(sentiments.get("positive") or 0)
    negatives = int(sentiments.get("negative") or 0)
    suggestions_count = int(sentiments.get("suggestion") or 0)

    lines = [
        f"<b>feedback last {hours}h</b>",
        "",
        f"total <b>{total}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· positive <b>{positives}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· negative <b>{negatives}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· suggestions <b>{suggestions_count}</b>",
    ]

    top_reasons = list(summary.get("top_reasons") or [])
    if top_reasons:
        reason_bits = [f"{safe_html(str(reason))} {count}" for reason, count in top_reasons[:5]]
        lines.append(f"top reasons: {', '.join(reason_bits)}")

    top_sources = list(summary.get("top_sources") or [])
    if top_sources:
        source_bits = [f"{safe_html(str(source))} {count}" for source, count in top_sources[:5]]
        lines.append(f"top sources: {', '.join(source_bits)}")

    recent = list(summary.get("recent") or [])
    if recent:
        lines.extend(["", "<b>recent feedback</b>"])
        for item in recent[:5]:
            ts = str(item.get("created_at") or "")
            stamp = safe_html(ts[11:16] if "T" in ts and len(ts) >= 16 else (ts[:16] or "recent"))
            reason = safe_html(str(item.get("reason") or "other"))
            sentiment = safe_html(str(item.get("sentiment") or "unknown"))
            preview = safe_html(str(item.get("reply_preview") or "")[:120] or "(no preview)")
            lines.append(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {stamp} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {sentiment}/{reason} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {preview}")

    suggestions = [item for item in list(summary.get("suggestions") or []) if item.get("improvement_text")]
    if suggestions:
        lines.extend(["", "<b>latest suggestions</b>"])
        for item in suggestions[:3]:
            suggestion = safe_html(str(item.get("improvement_text") or "")[:160])
            if suggestion:
                lines.append(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {suggestion}")

    if not recent and not suggestions:
        lines.extend(["", "no feedback captured in this window yet."])

    if summary.get("sampled"):
        lines.extend(["", f"<i>sample capped at {int(summary.get('limit') or 0)} rows.</i>"])

    return "\n".join(lines)


def _format_reply_stats_summary(summary: dict) -> str:
    hours = int(summary.get("hours") or 24)
    total = int(summary.get("total") or 0)
    touched = int(summary.get("touched") or 0)
    positives = int(summary.get("positive_feedback") or 0)
    negatives = int(summary.get("negative_feedback") or 0)
    suggestions_count = int(summary.get("suggestion_feedback") or 0)

    lines = [
        f"<b>reply stats last {hours}h</b>",
        "",
        f"replies <b>{total}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· touched <b>{touched}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· positive <b>{positives}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· negative <b>{negatives}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· suggestions <b>{suggestions_count}</b>",
        f"negative rate on touched replies: <b>{float(summary.get('negative_rate_pct') or 0.0):.1f}%</b>",
    ]

    top_routes = list(summary.get("top_routes") or [])
    if top_routes:
        route_bits = [f"{safe_html(str(route))} {count}" for route, count in top_routes[:5]]
        lines.append(f"top routes: {', '.join(route_bits)}")

    top_reply_kinds = list(summary.get("top_reply_kinds") or [])
    if top_reply_kinds:
        kind_bits = [f"{safe_html(str(kind))} {count}" for kind, count in top_reply_kinds[:5]]
        lines.append(f"top kinds: {', '.join(kind_bits)}")

    worst_routes = list(summary.get("worst_routes") or [])
    if worst_routes:
        lines.extend(["", "<b>worst routes</b>"])
        for route, neg, count, rate in worst_routes[:5]:
            lines.append(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {safe_html(str(route))} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {neg}/{count} negative ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {rate:.1f}%")

    recent = list(summary.get("recent") or [])
    if recent:
        lines.extend(["", "<b>recent replies</b>"])
        for item in recent[:5]:
            ts = str(item.get("created_at") or "")
            stamp = safe_html(ts[11:16] if "T" in ts and len(ts) >= 16 else (ts[:16] or "recent"))
            route = safe_html(str(item.get("route") or "unknown"))
            kind = safe_html(str(item.get("reply_kind") or "unknown"))
            preview = safe_html(str(item.get("reply_preview") or "")[:110] or "(no preview)")
            status = "negative" if item.get("has_negative") else ("positive" if item.get("has_positive") else "unrated")
            lines.append(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {stamp} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {route}/{kind} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {status} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {preview}")

    if not recent:
        lines.extend(["", "no reply analytics captured in this window yet."])

    if summary.get("sampled"):
        lines.extend(["", f"<i>sample capped at {int(summary.get('limit') or 0)} rows.</i>"])

    return "\n".join(lines)


def _format_quality_summary(summary: dict) -> str:
    hours = int(summary.get("hours") or 24)
    headline = dict(summary.get("headline") or {})
    top_reason = dict(headline.get("top_reason") or {})
    worst_route = dict(headline.get("worst_route") or {})

    lines = [
        f"<b>quality last {hours}h</b>",
        "",
        f"negative feedback <b>{int(headline.get('negative_feedback') or 0)}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· suggestions <b>{int(headline.get('suggestions') or 0)}</b> ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· reply negative rate <b>{float(headline.get('reply_negative_rate_pct') or 0.0):.1f}%</b>",
    ]

    if top_reason:
        lines.append(
            f"top complaint: <b>{safe_html(str(top_reason.get('reason') or 'other'))}</b> ({int(top_reason.get('count') or 0)})"
        )
    if worst_route:
        lines.append(
            "worst route: "
            f"<b>{safe_html(str(worst_route.get('route') or 'unknown'))}</b> "
            f"({int(worst_route.get('negative') or 0)}/{int(worst_route.get('total') or 0)} negative ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {float(worst_route.get('negative_rate_pct') or 0.0):.1f}%)"
        )

    feedback = dict(summary.get("feedback") or {})
    top_reasons = list(feedback.get("top_reasons") or [])
    if top_reasons:
        reason_bits = [f"{safe_html(str(reason))} {count}" for reason, count in top_reasons[:5]]
        lines.extend(["", f"top reasons: {', '.join(reason_bits)}"])

    replies = dict(summary.get("replies") or {})
    top_routes = list(replies.get("top_routes") or [])
    if top_routes:
        route_bits = [f"{safe_html(str(route))} {count}" for route, count in top_routes[:5]]
        lines.append(f"top routes: {', '.join(route_bits)}")

    bad_examples = list(summary.get("recent_negative_examples") or [])
    if bad_examples:
        lines.extend(["", "<b>recent bad examples</b>"])
        for item in bad_examples[:3]:
            ts = str(item.get("created_at") or "")
            stamp = safe_html(ts[11:16] if "T" in ts and len(ts) >= 16 else (ts[:16] or "recent"))
            reason = safe_html(str(item.get("reason") or "other"))
            preview = safe_html(str(item.get("preview") or "")[:120] or "(no preview)")
            lines.append(f"ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {stamp} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {reason} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {preview}")

    if not bad_examples:
        lines.extend(["", "no recent negative examples in this window."])

    return "\n".join(lines)


@router.message(Command("feedback"))
async def feedback_cmd(message: Message) -> None:
    await admin_command_flow.handle_feedback_command(message=message, deps=_admin_command_flow_dependencies())


@router.message(Command("replystats"))
async def replystats_cmd(message: Message) -> None:
    await admin_command_flow.handle_replystats_command(message=message, deps=_admin_command_flow_dependencies())


@router.message(Command("quality"))
async def quality_cmd(message: Message) -> None:
    await admin_command_flow.handle_quality_command(message=message, deps=_admin_command_flow_dependencies())


@router.message(Command("block"))
async def block_cmd(message: Message) -> None:
    await admin_command_flow.handle_block_command(message=message, deps=_admin_command_flow_dependencies())


@router.message(Command("unblock"))
async def unblock_cmd(message: Message) -> None:
    await admin_command_flow.handle_unblock_command(message=message, deps=_admin_command_flow_dependencies())


@router.message(Command("id"))
async def id_cmd(message: Message) -> None:
    if not message.from_user:
        await message.answer("couldn't read your user id from this update.")
        return
    await message.answer(
        f"your user id  <code>{message.from_user.id}</code>\n"
        f"this chat id  <code>{message.chat.id}</code>"
    )


@router.message(Command("settings"))
async def settings_cmd(message: Message) -> None:
    hub = _require_hub()
    settings = await hub.user_service.get_settings(message.chat.id)
    await message.answer(settings_text(settings), reply_markup=settings_menu(settings))


@router.message(Command("name"))
async def name_cmd(message: Message) -> None:
    """Set display name for memory (e.g. /name Alice)."""
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    name = (text or "").strip()[:64]
    await hub.user_service.update_settings(message.chat.id, {"display_name": name})
    await message.answer(f"Got it. I'll call you {name}." if name else "Display name cleared.")


@router.message(Command("goals"))
async def goals_cmd(message: Message) -> None:
    """Set trading goals for memory (e.g. /goals swing trading, low risk)."""
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    goals = (text or "").strip()[:300]
    await hub.user_service.update_settings(message.chat.id, {"trading_goals": goals})
    await message.answer("Goals saved. I'll keep that in mind." if goals else "Goals cleared.")


@router.message(Command("alpha"))
async def alpha_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_alpha_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("watch"))
async def watch_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_watch_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("price"))
async def price_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_price_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("chart"))
async def chart_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_chart_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("heatmap"))
async def heatmap_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_heatmap_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("rsi"))
async def rsi_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_rsi_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("ema"))
async def ema_cmd(message: Message) -> None:
    await shortcut_command_flow.handle_ema_command(message=message, deps=_dependency_factory().shortcut_command_flow())


@router.message(Command("watchlist"))
async def watchlist_cmd(message: Message) -> None:
    await alert_command_flow.handle_watchlist_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("news"))
async def news_cmd(message: Message) -> None:
    await alert_command_flow.handle_news_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("cycle"))
async def cycle_cmd(message: Message) -> None:
    await alert_command_flow.handle_cycle_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("scan"))
async def scan_cmd(message: Message) -> None:
    await alert_command_flow.handle_scan_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("alert"))
async def alert_cmd(message: Message) -> None:
    await alert_command_flow.handle_alert_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("alerts"))
async def alerts_cmd(message: Message) -> None:
    await alert_command_flow.handle_alerts_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("alertdel"))
async def alertdel_cmd(message: Message) -> None:
    await alert_command_flow.handle_alertdel_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("alertclear"))
async def alertclear_cmd(message: Message) -> None:
    await alert_command_flow.handle_alertclear_command(message=message, deps=_alert_command_flow_dependencies())


@router.message(Command("position"))
async def position_cmd(message: Message) -> None:
    await position_command_flow.handle_position_command(
        message=message,
        deps=_position_command_flow_dependencies(),
    )


@router.message(Command("journal"))
async def journal_cmd(message: Message) -> None:
    await data_account_command_flow.handle_journal_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("compare"))
async def compare_cmd(message: Message) -> None:
    await data_account_command_flow.handle_compare_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("report"))
async def report_cmd(message: Message) -> None:
    await data_account_command_flow.handle_report_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("export"))
async def export_cmd(message: Message) -> None:
    await data_account_command_flow.handle_export_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("mydata"))
async def mydata_cmd(message: Message) -> None:
    await data_account_command_flow.handle_mydata_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("deleteaccount"))
async def deleteaccount_cmd(message: Message) -> None:
    await data_account_command_flow.handle_deleteaccount_command(
        message=message,
        deps=_data_account_command_flow_dependencies(),
    )


@router.message(Command("tradecheck"))
async def tradecheck_cmd(message: Message) -> None:
    await trade_setup_command_flow.handle_tradecheck_command(
        message=message,
        deps=_trade_setup_command_flow_dependencies(),
    )


@router.message(Command("findpair"))
async def findpair_cmd(message: Message) -> None:
    await trade_setup_command_flow.handle_findpair_command(
        message=message,
        deps=_trade_setup_command_flow_dependencies(),
    )


@router.message(Command("setup"))
async def setup_cmd(message: Message) -> None:
    await trade_setup_command_flow.handle_setup_command(
        message=message,
        deps=_trade_setup_command_flow_dependencies(),
    )


@router.message(Command("margin"))
async def margin_cmd(message: Message) -> None:
    await trade_setup_command_flow.handle_margin_command(
        message=message,
        deps=_trade_setup_command_flow_dependencies(),
    )


@router.message(Command("pnl"))
async def pnl_cmd(message: Message) -> None:
    await trade_setup_command_flow.handle_pnl_command(
        message=message,
        deps=_trade_setup_command_flow_dependencies(),
    )


@router.message(Command("join"))
async def join_cmd(message: Message) -> None:
    await giveaway_command_flow.handle_join_command(
        message=message,
        deps=_giveaway_command_flow_dependencies(),
    )


@router.message(Command("giveaway"))
async def giveaway_cmd(message: Message) -> None:
    await giveaway_command_flow.handle_giveaway_command(
        message=message,
        deps=_giveaway_command_flow_dependencies(),
    )


@router.callback_query(F.data.startswith("followup:"))
async def followup_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: followup_callback_flow.handle_followup_callback(
            callback=callback,
            deps=_followup_callback_dependencies(),
        ),
    )


def _feedback_callback_dependencies() -> feedback_admin.FeedbackCallbackDependencies:
    return _dependency_factory().feedback_callback()



def _reaction_feedback_dependencies() -> feedback_admin.ReactionFeedbackDependencies:
    return _dependency_factory().reaction_feedback()


async def _notify_admins_negative_feedback(
    *,
    from_chat_id: int,
    from_username: str | None,
    reason: str,
    reply_preview: str,
    improvement_text: str | None = None,
) -> None:
    await feedback_helper_flow.notify_admins_negative_feedback(
        from_chat_id=from_chat_id,
        from_username=from_username,
        reason=reason,
        reply_preview=reply_preview,
        improvement_text=improvement_text,
        deps=_dependency_factory().feedback_helper(),
    )


@router.callback_query(F.data.startswith("feedback:"))
async def feedback_callback(callback: CallbackQuery) -> None:
    await feedback_admin.handle_feedback_callback(
        callback=callback,
        deps=_feedback_callback_dependencies(),
    )


@router.message_reaction()
async def message_reaction_handler(reaction_update: MessageReactionUpdated) -> None:
    await feedback_admin.handle_message_reaction(
        reaction_update=reaction_update,
        deps=_reaction_feedback_dependencies(),
    )


@router.callback_query(F.data.startswith("confirm:understood:"))
async def confirm_understood_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: followup_callback_flow.handle_confirm_understood_callback(callback=callback),
    )
@router.callback_query(F.data.startswith("confirm:clear_alerts:"))
async def confirm_clear_alerts_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: followup_callback_flow.handle_confirm_clear_alerts_callback(
            callback=callback,
            deps=_confirm_clear_alerts_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("cmd:"))
async def command_menu_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: command_menu_flow.handle_command_menu_callback(
            callback=callback,
            deps=_command_menu_flow_dependencies(),
        ),
    )


@router.callback_query(F.data.startswith("gw:"))
async def giveaway_menu_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: giveaway_menu_flow.handle_giveaway_menu_callback(
            callback=callback,
            deps=_giveaway_menu_flow_dependencies(),
        ),
    )


@router.callback_query(F.data.startswith("settings:"))
async def settings_callbacks(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: settings_callback_flow.handle_settings_callback(
            callback=callback,
            deps=_settings_callback_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("set_alert:"))
async def set_alert_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: analysis_detail_flow.handle_set_alert_callback(
            callback=callback,
            deps=_analysis_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("show_levels:"))
async def show_levels_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: analysis_detail_flow.handle_show_levels_callback(
            callback=callback,
            deps=_analysis_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("why:"))
async def why_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: analysis_detail_flow.handle_why_callback(
            callback=callback,
            deps=_analysis_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("refresh:"))
async def refresh_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: analysis_detail_flow.handle_refresh_callback(
            callback=callback,
            deps=_analysis_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("details:"))
async def details_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: analysis_detail_flow.handle_details_callback(
            callback=callback,
            deps=_analysis_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("derivatives:"))
async def derivatives_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: market_detail_flow.handle_derivatives_callback(
            callback=callback,
            deps=_market_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("catalysts:"))
async def catalysts_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: market_detail_flow.handle_catalysts_callback(
            callback=callback,
            deps=_market_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("backtest:"))
async def backtest_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: market_detail_flow.handle_backtest_callback(
            callback=callback,
            deps=_market_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("save_wallet:"))
async def save_wallet_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: market_detail_flow.handle_save_wallet_callback(
            callback=callback,
            deps=_market_detail_flow_dependencies(),
        ),
    )
@router.callback_query(F.data.startswith("quick:analysis:"))
async def quick_analysis_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_analysis_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("quick:analysis_tf:"))
async def quick_analysis_tf_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_analysis_tf_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("quick:chart:"))
async def quick_chart_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_chart_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("quick:heatmap:"))
async def quick_heatmap_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_heatmap_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("quick:rsi:"))
async def quick_rsi_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_rsi_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("quick:news:"))
async def quick_news_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_quick_news_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("define:"))
async def define_easter_egg_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_define_easter_egg_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.callback_query(F.data.startswith("top:"))
async def top_rsi_callback(callback: CallbackQuery) -> None:
    await _run_callback_handler(
        callback,
        lambda: quick_action_callback_flow.handle_top_rsi_callback(
            callback=callback,
            deps=_quick_action_callback_dependencies(),
        ),
    )

@router.message(F.text)
async def route_text(message: Message) -> None:
    await route_text_flow.handle_route_text(
        message,
        deps=_dependency_factory().route_text_flow(),
    )




