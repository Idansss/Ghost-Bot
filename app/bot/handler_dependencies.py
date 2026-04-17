from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from app.bot import (
    admin_command_flow,
    analysis_detail_flow,
    analysis_reply_flow,
    alert_command_flow,
    command_menu_flow,
    feedback_admin,
    feedback_helper_flow,
    free_text_flow,
    followup_callback_flow,
    giveaway_command_flow,
    giveaway_menu_flow,
    greeting_flow,
    llm_reply_flow,
    market_detail_flow,
    parsed_intent_executor,
    pre_route_state_executor,
    position_command_flow,
    quick_action_callback_flow,
    route_text_flow,
    routed_intent_executor,
    settings_callback_flow,
    shortcut_command_flow,
    source_query_flow,
    transport_runtime,
    trade_setup_command_flow,
    data_account_command_flow,
)


@dataclass(frozen=True)
class HandlerDependencyFactory:
    hub: Any
    settings_obj: Any
    logger: Any
    feedback_reason_labels: set[str]
    transport_runtime_deps: transport_runtime.TransportRuntimeDependencies
    source_query_deps: source_query_flow.SourceQueryFlowDependencies
    group_chat_deps: Any
    greeting_deps: greeting_flow.GreetingFlowDependencies
    helpers: Any

    def llm_reply_flow(self) -> llm_reply_flow.LlmReplyFlowDependencies:
        return llm_reply_flow.LlmReplyFlowDependencies(
            hub=self.hub,
            openai_chat_history_turns=int(self.settings_obj.openai_chat_history_turns),
            openai_max_output_tokens=int(self.settings_obj.openai_max_output_tokens),
            openai_temperature=float(self.settings_obj.openai_temperature),
            bot_meta_re=self.helpers.bot_meta_re,
            build_communication_memory_block=self.helpers.build_communication_memory_block,
            format_market_context=self.helpers.format_market_context,
            safe_html=self.helpers.safe_html,
            ghost_persona=self.helpers.ghost_persona,
            try_answer_definition=self.helpers.try_answer_definition,
            try_answer_howto=self.helpers.try_answer_howto,
        )

    def analysis_reply_flow(self) -> analysis_reply_flow.AnalysisReplyFlowDependencies:
        return analysis_reply_flow.AnalysisReplyFlowDependencies(
            format_as_ghost=self.helpers.format_as_ghost,
            llm_analysis_reply=self.helpers.llm_analysis_reply,
            trade_plan_template=self.helpers.trade_plan_template,
            analysis_progressive_menu=self.helpers.analysis_progressive_menu,
            pause=self.helpers.pause,
        )

    def routed_intent(self) -> routed_intent_executor.RoutedIntentDependencies:
        return routed_intent_executor.RoutedIntentDependencies(
            hub=self.hub,
            openai_router_min_confidence=self.settings_obj.openai_router_min_confidence,
            bot_meta_re=self.helpers.bot_meta_re,
            try_answer_howto=self.helpers.try_answer_howto,
            llm_fallback_reply=self.helpers.llm_fallback_reply,
            llm_market_chat_reply=self.helpers.llm_market_chat_reply,
            send_llm_reply=self.helpers.send_llm_reply,
            as_int=self.helpers.as_int,
            as_float=self.helpers.as_float,
            as_float_list=self.helpers.as_float_list,
            extract_symbol=self.helpers.extract_symbol,
            normalize_symbol_value=self.helpers.normalize_symbol_value,
            analysis_timeframes_from_settings=self.helpers.analysis_timeframes_from_settings,
            parse_int_list=self.helpers.parse_int_list,
            append_last_symbol=self.helpers.append_last_symbol,
            remember_analysis_context=self.helpers.remember_analysis_context,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            render_analysis_text=self.helpers.render_analysis_text,
            send_ghost_analysis=self.helpers.send_ghost_analysis,
            safe_exc=self.helpers.safe_exc,
            parse_duration_to_seconds=self.helpers.parse_duration_to_seconds,
            trade_math_payload=self.helpers.trade_math_payload,
            feature_flags_set=self.settings_obj.feature_flags_set,
        )

    def pre_route_state(self) -> pre_route_state_executor.PreRouteStateDependencies:
        helper_deps = self.feedback_helper()
        return pre_route_state_executor.PreRouteStateDependencies(
            hub=self.hub,
            as_int=self.helpers.as_int,
            dispatch_command_text=self.helpers.dispatch_command_text,
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
            get_cmd_wizard=self.helpers.get_cmd_wizard,
            clear_cmd_wizard=self.helpers.clear_cmd_wizard,
            get_wizard=self.helpers.get_wizard,
            set_wizard=self.helpers.set_wizard,
            clear_wizard=self.helpers.clear_wizard,
            parse_timestamp=self.helpers.parse_timestamp,
            save_trade_check=self.helpers.save_trade_check,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            get_pending_alert=self.helpers.get_pending_alert,
            clear_pending_alert=self.helpers.clear_pending_alert,
            trade_verification_template=self.helpers.trade_verification_template,
            giveaway_menu=self.helpers.giveaway_menu,
            alert_created_menu=self.helpers.alert_created_menu,
        )

    def free_text_flow(self) -> free_text_flow.FreeTextFlowDependencies:
        return free_text_flow.FreeTextFlowDependencies(
            hub=self.hub,
            parse_message=self.helpers.parse_message,
            openai_chat_mode=self.helpers.openai_chat_mode,
            route_free_text=self.helpers.route_free_text,
            send_llm_reply=self.helpers.send_llm_reply,
            get_chat_history=self.helpers.get_chat_history,
            dispatch_command_text=self.helpers.dispatch_command_text,
            recent_analysis_context=self.helpers.recent_analysis_context,
            looks_like_analysis_followup=self.helpers.looks_like_analysis_followup,
            llm_followup_reply=self.helpers.llm_followup_reply,
            llm_market_chat_reply=self.helpers.llm_market_chat_reply,
            llm_route_message=self.helpers.llm_route_message,
            handle_routed_intent=self.helpers.handle_routed_intent,
            handle_parsed_intent=self.helpers.handle_parsed_intent,
            is_definition_question=self.helpers.is_definition_question,
            is_likely_english_phrase=self.helpers.is_likely_english_phrase,
            extract_action_symbol_hint=self.helpers.extract_action_symbol_hint,
            clarifying_question=self.helpers.clarifying_question,
            smart_action_menu=self.helpers.smart_action_menu,
            analysis_symbol_followup_kb=self.helpers.analysis_symbol_followup_kb,
            define_keyboard=self.helpers.define_keyboard,
            pause=self.helpers.pause,
        )

    def command_menu_flow(self) -> command_menu_flow.CommandMenuFlowDependencies:
        return command_menu_flow.CommandMenuFlowDependencies(
            hub=self.hub,
            dispatch_command_text=self.helpers.dispatch_command_text,
            run_with_typing_lock=self.helpers.run_with_typing_lock,
            set_cmd_wizard=self.helpers.set_cmd_wizard,
            set_wizard=self.helpers.set_wizard,
            as_int=self.helpers.as_int,
            alpha_quick_menu=self.helpers.alpha_quick_menu,
            watch_quick_menu=self.helpers.watch_quick_menu,
            chart_quick_menu=self.helpers.chart_quick_menu,
            heatmap_quick_menu=self.helpers.heatmap_quick_menu,
            rsi_quick_menu=self.helpers.rsi_quick_menu,
            ema_quick_menu=self.helpers.ema_quick_menu,
            news_quick_menu=self.helpers.news_quick_menu,
            alert_quick_menu=self.helpers.alert_quick_menu,
            findpair_quick_menu=self.helpers.findpair_quick_menu,
            setup_quick_menu=self.helpers.setup_quick_menu,
            scan_quick_menu=self.helpers.scan_quick_menu,
            giveaway_menu=self.helpers.giveaway_menu,
            rsi_scan_template=self.helpers.rsi_scan_template,
            logger=self.logger,
        )

    def giveaway_menu_flow(self) -> giveaway_menu_flow.GiveawayMenuFlowDependencies:
        return giveaway_menu_flow.GiveawayMenuFlowDependencies(
            hub=self.hub,
            run_with_typing_lock=self.helpers.run_with_typing_lock,
            set_cmd_wizard=self.helpers.set_cmd_wizard,
            as_int=self.helpers.as_int,
            giveaway_duration_menu=self.helpers.giveaway_duration_menu,
            giveaway_winners_menu=self.helpers.giveaway_winners_menu,
            giveaway_status_template=self.helpers.giveaway_status_template,
        )

    def followup_callback(self) -> followup_callback_flow.FollowupCallbackDependencies:
        return followup_callback_flow.FollowupCallbackDependencies(
            hub=self.hub,
            sanitize_html=self.helpers.sanitize_html,
            llm_reply_keyboard=self.helpers.llm_reply_keyboard,
        )

    def confirm_clear_alerts(self) -> followup_callback_flow.ConfirmClearAlertsDependencies:
        return followup_callback_flow.ConfirmClearAlertsDependencies(
            clear_user_alerts=self.hub.alerts_service.clear_user_alerts,
        )

    def settings_callback(self) -> settings_callback_flow.SettingsCallbackDependencies:
        return settings_callback_flow.SettingsCallbackDependencies(
            get_user_settings=self.hub.user_service.get_settings,
            update_user_settings=self.hub.user_service.update_settings,
            settings_text=self.helpers.settings_text,
            settings_menu=self.helpers.settings_menu,
        )

    def analysis_detail_flow(self) -> analysis_detail_flow.AnalysisDetailFlowDependencies:
        return analysis_detail_flow.AnalysisDetailFlowDependencies(
            hub=self.hub,
            set_pending_alert=self.helpers.set_pending_alert,
            run_with_typing_lock=self.helpers.run_with_typing_lock,
            analysis_timeframes_from_settings=self.helpers.analysis_timeframes_from_settings,
            parse_int_list=self.helpers.parse_int_list,
            remember_analysis_context=self.helpers.remember_analysis_context,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            render_analysis_text=self.helpers.render_analysis_text,
            send_ghost_analysis=self.helpers.send_ghost_analysis,
        )

    def market_detail_flow(self) -> market_detail_flow.MarketDetailFlowDependencies:
        return market_detail_flow.MarketDetailFlowDependencies(
            hub=self.hub,
            feature_flags_set=self.settings_obj.feature_flags_set,
            run_with_typing_lock=self.helpers.run_with_typing_lock,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
        )

    def quick_action_callback(self) -> quick_action_callback_flow.QuickActionCallbackDependencies:
        return quick_action_callback_flow.QuickActionCallbackDependencies(
            hub=self.hub,
            run_with_typing_lock=self.helpers.run_with_typing_lock,
            analysis_timeframes_from_settings=self.helpers.analysis_timeframes_from_settings,
            parse_int_list=self.helpers.parse_int_list,
            remember_analysis_context=self.helpers.remember_analysis_context,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            render_analysis_text=self.helpers.render_analysis_text,
            send_ghost_analysis=self.helpers.send_ghost_analysis,
            set_pending_alert=self.helpers.set_pending_alert,
            as_int=self.helpers.as_int,
            rsi_scan_template=self.helpers.rsi_scan_template,
            news_template=self.helpers.news_template,
        )

    def admin_command_flow(self) -> admin_command_flow.AdminCommandFlowDependencies:
        return admin_command_flow.AdminCommandFlowDependencies(
            hub=self.hub,
            is_bot_admin=self.helpers.is_bot_admin,
            admin_ids_list=self.settings_obj.admin_ids_list,
            format_feedback_summary=self.helpers.format_feedback_summary,
            format_reply_stats_summary=self.helpers.format_reply_stats_summary,
            format_quality_summary=self.helpers.format_quality_summary,
            abuse_block_ttl_sec=self.settings_obj.abuse_block_ttl_sec,
            abuse_block_key=transport_runtime.abuse_block_key,
            abuse_strike_key=transport_runtime.abuse_strike_key,
            record_abuse=self.helpers.record_abuse,
        )

    def shortcut_command_flow(self) -> shortcut_command_flow.ShortcutCommandFlowDependencies:
        return shortcut_command_flow.ShortcutCommandFlowDependencies(
            hub=self.hub,
            dispatch_command_text=self.helpers.dispatch_command_text,
            alpha_quick_menu=self.helpers.alpha_quick_menu,
            watch_quick_menu=self.helpers.watch_quick_menu,
            chart_quick_menu=self.helpers.chart_quick_menu,
            heatmap_quick_menu=self.helpers.heatmap_quick_menu,
            rsi_quick_menu=self.helpers.rsi_quick_menu,
            ema_quick_menu=self.helpers.ema_quick_menu,
            as_int=self.helpers.as_int,
        )

    def alert_command_flow(self) -> alert_command_flow.AlertCommandFlowDependencies:
        return alert_command_flow.AlertCommandFlowDependencies(
            hub=self.hub,
            wallet_scan_limit_per_hour=self.settings_obj.wallet_scan_limit_per_hour,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            watchlist_template=self.helpers.watchlist_template,
            news_template=self.helpers.news_template,
            cycle_template=self.helpers.cycle_template,
            wallet_scan_template=self.helpers.wallet_scan_template,
            wallet_actions=self.helpers.wallet_actions,
            scan_quick_menu=self.helpers.scan_quick_menu,
            news_quick_menu=self.helpers.news_quick_menu,
            alert_quick_menu=self.helpers.alert_quick_menu,
            simple_followup=self.helpers.simple_followup,
            alert_created_menu=self.helpers.alert_created_menu,
        )

    def position_command_flow(self) -> position_command_flow.PositionCommandFlowDependencies:
        return position_command_flow.PositionCommandFlowDependencies(
            hub=self.hub,
            feature_flags_set=self.settings_obj.feature_flags_set,
        )

    def data_account_command_flow(self) -> data_account_command_flow.DataAccountCommandFlowDependencies:
        return data_account_command_flow.DataAccountCommandFlowDependencies(
            hub=self.hub,
            feature_flags_set=self.settings_obj.feature_flags_set,
            safe_exc=self.helpers.safe_exc,
            buffered_input_file_cls=self.helpers.buffered_input_file_cls,
        )

    def trade_setup_command_flow(self) -> trade_setup_command_flow.TradeSetupCommandFlowDependencies:
        return trade_setup_command_flow.TradeSetupCommandFlowDependencies(
            set_wizard=self.helpers.set_wizard,
            dispatch_command_text=self.helpers.dispatch_command_text,
            findpair_quick_menu=self.helpers.findpair_quick_menu,
            setup_quick_menu=self.helpers.setup_quick_menu,
        )

    def giveaway_command_flow(self) -> giveaway_command_flow.GiveawayCommandFlowDependencies:
        return giveaway_command_flow.GiveawayCommandFlowDependencies(
            hub=self.hub,
            safe_exc=self.helpers.safe_exc,
            parse_duration_to_seconds=self.helpers.parse_duration_to_seconds,
            giveaway_status_template=self.helpers.giveaway_status_template,
            giveaway_menu=self.helpers.giveaway_menu,
        )

    def feedback_helper(self) -> feedback_helper_flow.FeedbackHelperDependencies:
        return feedback_helper_flow.FeedbackHelperDependencies(
            hub=self.hub,
            admin_ids_list=self.settings_obj.admin_ids_list,
            logger=self.logger,
            record_feedback_metric=self.helpers.record_feedback_metric,
            allowed_reasons=self.feedback_reason_labels,
        )

    def feedback_callback(self) -> feedback_admin.FeedbackCallbackDependencies:
        return feedback_helper_flow.build_feedback_callback_dependencies(
            helper_deps=self.feedback_helper(),
            acquire_callback_once=self.helpers.acquire_callback_once,
            feedback_reason_kb=self.helpers.feedback_reason_kb,
            get_user_settings=self.hub.user_service.get_settings,
            update_user_settings=self.hub.user_service.update_settings,
        )

    def reaction_feedback(self) -> feedback_admin.ReactionFeedbackDependencies:
        return feedback_helper_flow.build_reaction_feedback_dependencies(
            helper_deps=self.feedback_helper(),
        )

    def parsed_intent(self) -> parsed_intent_executor.ParsedIntentDependencies:
        return parsed_intent_executor.ParsedIntentDependencies(
            hub=self.hub,
            wallet_scan_limit_per_hour=self.settings_obj.wallet_scan_limit_per_hour,
            maybe_send_market_warning=partial(greeting_flow.maybe_send_market_warning, deps=self.greeting_deps),
            analysis_timeframes_from_settings=self.helpers.analysis_timeframes_from_settings,
            parse_int_list=self.helpers.parse_int_list,
            append_last_symbol=self.helpers.append_last_symbol,
            remember_analysis_context=self.helpers.remember_analysis_context,
            remember_source_context=partial(source_query_flow.remember_source_context, deps=self.source_query_deps),
            render_analysis_text=self.helpers.render_analysis_text,
            send_ghost_analysis=self.helpers.send_ghost_analysis,
            as_float=self.helpers.as_float,
            trade_math_payload=self.helpers.trade_math_payload,
            llm_fallback_reply=self.helpers.llm_fallback_reply,
            safe_exc=self.helpers.safe_exc,
            parse_duration_to_seconds=self.helpers.parse_duration_to_seconds,
            save_trade_check=self.helpers.save_trade_check,
        )

    def route_text_flow(self) -> route_text_flow.RouteTextFlowDependencies:
        return route_text_flow.RouteTextFlowDependencies(
            hub=self.hub,
            logger=self.logger,
            record_abuse=self.helpers.record_abuse,
            is_blocked_subject=partial(transport_runtime.is_blocked_subject, deps=self.transport_runtime_deps),
            blocked_notice_ttl=partial(transport_runtime.blocked_notice_ttl, deps=self.transport_runtime_deps),
            is_group_admin=partial(self.helpers.group_is_admin, deps=self.group_chat_deps),
            set_group_free_talk=partial(self.helpers.set_group_free_talk, deps=self.group_chat_deps),
            group_free_talk_enabled=partial(self.helpers.group_free_talk_enabled, deps=self.group_chat_deps),
            mentions_bot=self.helpers.mentions_bot,
            strip_bot_mention=self.helpers.strip_bot_mention,
            is_reply_to_bot=partial(self.helpers.is_reply_to_bot, bot_id=self.hub.bot.id),
            looks_like_clear_intent=partial(self.helpers.looks_like_clear_intent, deps=self.group_chat_deps),
            is_source_query=source_query_flow.is_source_query,
            source_reply_for_chat=partial(source_query_flow.source_reply_for_chat, deps=self.source_query_deps),
            acquire_message_once=partial(transport_runtime.acquire_message_once, deps=self.transport_runtime_deps),
            check_request_limit=partial(transport_runtime.check_request_limit, deps=self.transport_runtime_deps),
            record_strike_and_maybe_block=partial(transport_runtime.record_strike_and_maybe_block, deps=self.transport_runtime_deps),
            greeting_re=self.helpers.greeting_re,
            choose_reply=self.helpers.choose_reply,
            gn_replies=self.helpers.gn_replies,
            market_aware_gm_reply=partial(greeting_flow.market_aware_gm_reply, deps=self.greeting_deps),
            chat_lock=self.transport_runtime_deps.chat_lock,
            typing_loop=transport_runtime.typing_loop,
            handle_pre_route_state=self.helpers.handle_pre_route_state,
            handle_free_text_flow=self.helpers.handle_free_text_flow,
            safe_exc=self.helpers.safe_exc,
            now_utc=self.helpers.now_utc,
            blocked_message=self.helpers.blocked_message,
            blocked_rate_limit_message=self.helpers.blocked_rate_limit_message,
            rate_limit_notice=self.helpers.rate_limit_notice,
            plain_text_prompt=self.helpers.plain_text_prompt,
            busy_notice=self.helpers.busy_notice,
        )
