#include "server-task.h"
#include "server-chat.h"

json result_timings::to_json() const {
    json base = {
        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},

        {"n_ctx",           n_ctx},
        {"n_past",           n_past},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
        if (!draft_n_by_depth.empty()) {
            json by_depth = json::array();
            for (size_t i = 0; i < draft_n_by_depth.size(); ++i) {
                if (draft_n_by_depth[i] <= 0) {
                    continue;
                }
                const int32_t accepted = i < draft_n_accepted_by_depth.size()
                    ? draft_n_accepted_by_depth[i] : 0;
                by_depth.push_back({
                    {"depth",              (int32_t) i + 1},
                    {"draft_n",            draft_n_by_depth[i]},
                    {"draft_n_accepted",   accepted},
                });
            }
            if (!by_depth.empty()) {
                base["draft_by_depth"] = by_depth;
            }
        }
    }

    return base;
}


//json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
//    // non-OAI-compat JSON
//    json res = json{
//        {"index",            index},
//        {"content",          content},
//        {"tokens",           tokens},
//        {"stop",             false},
//        {"id_slot",          id_multi},
//        {"tokens_predicted", n_decoded},
//        {"tokens_evaluated", n_prompt_tokens},
//    };
//    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
//    if (timings.prompt_n > 0) {
//        res.push_back({ "timings", timings.to_json() });
//    }
//    if (!probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return res;
//}

//json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
//    json res = json{
//        {"index",               index},
//        {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
//        {"tokens",              stream ? std::vector<llama_token> {} : tokens},
//        {"id_slot",             id_multi},
//        {"stop",                true},
//        {"model",               oaicompat_model},
//        {"tokens_predicted",    n_decoded},
//        {"tokens_evaluated",    n_prompt_tokens},
//        //{"generation_settings", default_generation_settings_for_props.to_json()},
//        {"prompt",              prompt},
//        {"has_new_line",        has_new_line},
//        {"truncated",           truncated},
//        //{"stop_type",           stop_type_to_str(STOP_TYPE_EOS)},
//        {"stopping_word",       stopping_word},
//        {"tokens_cached",       n_tokens_cached},
//        {"timings",             timings.to_json()},
//    };
//    if (!stream && !probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
//}

json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_partial::to_json_oaicompat_partial() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_partial();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json{
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}


json server_task_result_cmpl_final::to_json_oaicompat_final() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat_partial() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json& delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    };
    // We have to send an initial update to conform to openai behavior
    if (first) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {        
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

        if (probs_output.size() > 0) {
            deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            deltas[deltas.size() - 1].push_back({ "timings", timings.to_json() });
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp_partial() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json{
            {"event", "response.created"},
            {"data", json{
                {"type", "response.created"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json{
            {"event", "response.in_progress"},
            {"data", json{
                {"type", "response.in_progress"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!oai_resp_thinking_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                oai_resp_thinking_block_started = true;
            }
            events.push_back(json{
                {"event", "response.reasoning_text.delta"},
                {"data", json{
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!oai_resp_text_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json{
                    {"event", "response.content_part.added"},
                    {"data", json{
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json{
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                oai_resp_text_block_started = true;
            }
            events.push_back(json{
                {"event", "response.output_text.delta"},
                {"data", json{
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json{
                {"event", "response.output_item.added"},
                {"data", json{
                    {"type",  "response.output_item.added"},
                    {"item", json{
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json{
                {"event", "response.function_call_arguments.delta"},
                {"data", json{
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    return events;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_final() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }


    json choice{
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json{
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto& diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", 0},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"object",             "chat.completion.chunk"},
        });
    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
            });
    }
    if (timings.prompt_n >= 0) {
        deltas.back().push_back({ "timings", timings.to_json() });
    }
    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat_final();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_final() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (!msg.reasoning_content.empty()) {
        output.push_back(json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (!msg.content.empty()) {
        output.push_back(json{
            {"content", json::array({json{
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     oai_resp_message_id},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json{
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> events;
    std::vector<json> output;

    if (!oaicompat_msg.reasoning_content.empty()) {
        const json output_item = json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    if (!oaicompat_msg.content.empty()) {
        events.push_back(json{
            {"event", "response.output_text.done"},
            {"data", json{
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content},
            }},
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content},
        };

        events.push_back(json{
            {"event", "response.content_part.done"},
            {"data", json{
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part},
            }},
        });

        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        };
        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    events.push_back(json{
        {"event", "response.completed"},
        {"data", json{
            {"type", "response.completed"},
            {"response", json{
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }},
    });

    return events;
}

json server_task_result_cmpl_final::to_json_anthropic_final() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
            });
    }

    for (const auto& tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        }
        catch (const std::exception&) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_thinking_block_started ? 1 : 0;

    bool thinking_block_started = anthropic_thinking_block_started;
    bool text_block_started = anthropic_text_block_started;
    std::set<size_t> tool_calls_started;

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto& full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name},
                            // as Anthropic's API opens the block; a client may refuse a block without it
                            {"input", json::object()}
                        }}
                    }}
                    });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (thinking_block_started) {
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
            });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
            });
    }

    if (text_block_started) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
            });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
            });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
        });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
        });

    // extra fields for debugging purposes
    if (verbose && !events.empty()) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_final();
    }
    // Don't add timings for Anthropic API (breaks spec compliance)
    if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_anthropic_partial() {
    json events = json::array();
    bool first = n_decoded == 1;

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_has_reasoning ? 1 : 0;

    bool thinking_started = anthropic_thinking_block_started;
    bool text_started = anthropic_text_block_started;

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name},
                            // as Anthropic's API opens the block; a client may refuse a block without it
                            {"input", json::object()}
                        }}
                    }}
                    });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (verbose && !events.empty() && first) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_partial();
    }

    if (timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    //if (is_progress && !events.empty()) {
    //    events.back()["data"]["prompt_progress"] = progress.to_json();
    //}

    return events;
}


size_t server_prompt::size() const {
    size_t res = data.size() + file.state_size + file.ckpt_size;

    for (const auto& checkpoint : checkpoints) {
        res += checkpoint.size();
    }

    return res;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.n_tokens();
    }
    return res;

}

bool server_prompt_cache::load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot, float min_reusable_fraction) {
    thinking_tokens think_tokens;
    for (auto it = states.begin(); it != states.end(); ++it) {
        think_tokens = it->think_tokens;
        break;
    }
    server_tokens prompt_tokens;
    server_tokens tokens_new_ex;
    if (think_tokens.exclude) {
        prompt_tokens = prompt.tokens.get_tokens_exclude_think(ctx, think_tokens);
        tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx, think_tokens);
    }
    else {
        prompt_tokens = prompt.tokens.clone();
        tokens_new_ex = tokens_new.clone();
    }
    const auto lcp_best = prompt_tokens.get_common_prefix(ctx, tokens_new_ex);
    float f_keep_best = float(lcp_best.second) / prompt_tokens.size();
    float sim_best = prompt_tokens.get_tokens_similarity(ctx, tokens_new_ex, prompt.n_kept_prompt, prompt.n_discarded_prompt);
    LLAMA_LOG_INFO(" - looking for better prompt, base f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        server_tokens tokens;
        if (think_tokens.exclude) {
            tokens = it->tokens.get_tokens_exclude_think(ctx, think_tokens);
        }
        else {
            tokens = it->tokens.clone();
        }
        const auto lcp_cur = tokens.get_common_prefix(ctx, tokens_new_ex);
        const float f_keep_cur = float(lcp_cur.first) / tokens.size();
        if (f_keep_cur < min_reusable_fraction) {
            continue;
        }
        const float sim_cur = tokens.get_tokens_similarity(ctx, tokens_new_ex, it->n_kept_prompt, it->n_discarded_prompt);
        if (sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best = sim_cur;
            it_best = it;
        }
    }

    if (it_best != states.end() && it_best->in_file()) {
        LLAMA_LOG_INFO(" - found better prompt with f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d, in %s\n", f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt, it_best->file.path.c_str());
        const int64_t t_start = ggml_time_us();
        const size_t size = it_best->file.state_size;
        // The checkpoints and the mapping first: if the file cannot be read, the context was not touched and the
        // slot goes on with what it has. The file is given up after three such failures.
        std::vector<server_prompt_store::ckpt_out> ckpts;
        const uint8_t * state = nullptr;
        if (store->read_checkpoints(it_best->file, ckpts)) {
            state = store->map_state(it_best->file);
        }
        if (!state) {
            const bool give_up = ++it_best->n_failed >= 3;
            LLAMA_LOG_WARN(" - %s%s\n", store->error.c_str(), give_up ? ": the prompt is dropped" : "");
            if (give_up) {
                erase(it_best);
            }
            return false;
        }
        const size_t n = llama_state_seq_set_data(ctx, state, size, id_slot, 0);
        store->unmap_state(it_best->file, state);
        const llama_pos pos_max = llama_kv_cache_seq_pos_max(ctx, id_slot);
        // the checkpoints the slot had are those of the prompt it leaves
        for (auto & c : prompt.checkpoints) {
            if (!c.spill_path.empty()) {
                remove(c.spill_path.c_str());
            }
        }
        prompt.checkpoints.clear();
        prompt.data.clear();
        if (n != size || (it_best->file_pos_max >= 0 && pos_max != it_best->file_pos_max)) {
            // A file of another build or of other settings can fail here, with the sequence in an unknown
            // state: the slot starts from nothing, and the file is not tried again.
            LLAMA_LOG_WARN(" - the state of %s was not restored (%zu of %zu bytes, last position %d, expected %d): the prompt is dropped and the slot is emptied\n",
                it_best->file.path.c_str(), n, size, (int)pos_max, (int)it_best->file_pos_max);
            llama_kv_cache_seq_rm(ctx, id_slot, -1, -1);
            prompt.tokens.keep_first(0);
            prompt.n_kept_prompt = 0;
            prompt.n_discarded_prompt = 0;
            if (store_settings.differs(it_best->file_settings)) {
                // written by a server that was set up in another way, where it may still be good: the file
                // stays, and this server does not try it again
                const auto & f = it_best->file_settings;
                LLAMA_LOG_WARN(" - it was written by a server with a context of %u tokens, batches of %u and %u slot(s), this one has %u, %u and %u: the file is left in %s\n",
                    f.n_ctx, f.n_ubatch, f.n_parallel, store_settings.n_ctx, store_settings.n_ubatch, store_settings.n_parallel, store->dir.c_str());
                states.erase(it_best);
            } else {
                erase(it_best);
            }
            return false;
        }
        // The slot gets the prompt and its checkpoints. The file stays in the cache, held by the slot, until the
        // slot writes a newer state of this conversation: a server that dies meanwhile finds it at its next start.
        for (auto & other : states) {
            if (other.held_by == id_slot) {
                other.held_by = -1;
            }
        }
        it_best->held_by  = id_slot;
        it_best->n_failed = 0;
        prompt.tokens             = it_best->tokens.clone();
        prompt.n_kept_prompt      = it_best->n_kept_prompt;
        prompt.n_discarded_prompt = it_best->n_discarded_prompt;
        prompt.think_tokens       = it_best->think_tokens;
        for (auto & c : ckpts) {
            auto & cur = prompt.checkpoints.emplace_back();
            cur.pos_min    = c.pos_min;
            cur.pos_max    = c.pos_max;
            cur.n_tokens   = c.n_tokens;
            cur.prompt_end = c.prompt_end;
            cur.data       = std::move(c.data);
        }
        LLAMA_LOG_INFO(" - read %.3f MiB and %zu checkpoints from the file in %.2f ms\n", size / (1024.0 * 1024.0), prompt.checkpoints.size(), (ggml_time_us() - t_start) / 1000.0);
    }
    else if (it_best != states.end()) {
        LLAMA_LOG_INFO(" - found better prompt with f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt);
        const size_t size = it_best->data.size();
        const size_t n = llama_state_seq_set_data(ctx, it_best->data.data(), size, id_slot, 0);
        if (n != size) {
            LLAMA_LOG_INFO("failed to restore state with size %zu\n", size);
            return false;
        }

        it_best->data.clear();
        it_best->data.shrink_to_fit();

        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

bool server_prompt_cache::forget_contained(const server_prompt& prompt) {
    for (auto it = states.begin(); it != states.end();) {
        auto tokens_ctx_shift = prompt.tokens.clone();  // copy cache tokens
        tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
        auto prefix = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
        const size_t len = prefix.first;
        const size_t len_prompt = prefix.second;
        // first check if the current state is contained fully in the cache
        if (len_prompt == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return false;
        }
        // next, remove any cached prompts that are fully contained in the current prompt
        else if (len == it->tokens.size()) {
            LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", (int)len);
            it = erase(it);
        }
        else {
            ++it;
        }
    }
    return true;
}

std::list<server_prompt>::iterator server_prompt_cache::erase(std::list<server_prompt>::iterator it) {
    if (store && it->in_file()) {
        store->remove(it->file);
    }
    return states.erase(it);
}

void server_prompt_cache::clear() {
    while (!states.empty()) {
        erase(states.begin());
    }
}

bool server_prompt_cache::open_store(const std::string & dir, const std::string & model_fingerprint,
        const std::vector<std::pair<std::string, uint32_t>> & earlier, const server_prompt_settings & settings) {
    auto s = std::make_unique<server_prompt_store>();
    std::vector<uint64_t> also;
    for (const auto & f : earlier) {
        also.push_back(server_prompt_store::hash(f.first));
    }
    if (!s->open(dir, server_prompt_store::hash(model_fingerprint), also)) {
        LLAMA_LOG_WARN("prompt cache: %s: it stays in the memory of the server\n", s->error.c_str());
        return false;
    }
    size_t n_other = 0, n_bad = 0, n_unread = 0, n_long = 0, n_other_ctx = 0;
    auto found = s->scan(n_other, n_bad);
    store = std::move(s);
    store_settings = settings;
    const uint32_t n_ctx = settings.n_ctx, n_ctx_slot = settings.n_ctx_slot();
    for (auto & e : found) {
        server_prompt cur;
        try {
            const json meta = json::parse(e.meta);
            cur.from_json(meta);
            const json think = meta.value("think", json::object());
            cur.think_tokens.exclude = think.value("exclude", cur.think_tokens.exclude);
            cur.think_tokens.begin   = think.value("begin",   cur.think_tokens.begin);
            cur.think_tokens.end     = think.value("end",     cur.think_tokens.end);
            cur.file_pos_max         = meta.value<llama_pos>("pos_max", -1);
            cur.file_settings.n_ctx      = meta.value<uint32_t>("n_ctx", 0);
            cur.file_settings.n_ubatch   = meta.value<uint32_t>("n_ubatch", 0);
            cur.file_settings.n_parallel = meta.value<uint32_t>("n_parallel", 0);
        }
        catch (const std::exception &) {
            ++n_unread;
            continue;
        }
        // a file of an earlier version does not say its context size: its fingerprint does
        for (size_t i = 0; cur.file_settings.n_ctx == 0 && i < also.size(); ++i) {
            if (also[i] == e.fingerprint) {
                cur.file_settings.n_ctx = earlier[i].second;
            }
        }
        // A prompt that is longer than the context of a slot cannot come back here: its file is left for a
        // server with a larger context.
        if (std::max<int64_t>(cur.n_tokens(), (int64_t)cur.file_pos_max + 1) > (int64_t)n_ctx_slot) {
            ++n_long;
            continue;
        }
        if (cur.file_settings.n_ctx != 0 && cur.file_settings.n_ctx != n_ctx) {
            ++n_other_ctx;
        }
        e.meta.clear();
        cur.file = std::move(e);
        states.push_back(std::move(cur));
    }
    LLAMA_LOG_INFO("prompt cache: kept in %s, where %zu prompt(s) of this model were found (%.3f MiB)\n",
        store->dir.c_str(), states.size(), size() / (1024.0 * 1024.0));
    if (n_other_ctx > 0) {
        LLAMA_LOG_INFO("prompt cache: %zu of them were written with another context size than the %u tokens of this server: they fit, and are taken back like the others\n",
            n_other_ctx, n_ctx);
    }
    if (n_long > 0) {
        LLAMA_LOG_WARN("prompt cache: %zu prompt(s) of this model are longer than the %u tokens of a slot and are left in %s for a server with a larger context: remove them if they are not wanted\n",
            n_long, n_ctx_slot, store->dir.c_str());
    }
    if (n_other + n_bad + n_unread > 0) {
        LLAMA_LOG_WARN("prompt cache: %zu file(s) of another model and %zu that cannot be read are left in %s: remove them if they are not wanted\n",
            n_other, n_bad + n_unread, store->dir.c_str());
    }
    return true;
}

bool server_prompt_cache::starts_alike(const server_prompt & cached, const server_prompt & prompt) const {
    if (cached.tokens.size() == 0) {
        return false;
    }
    const thinking_tokens & think = cached.think_tokens;
    if (think.exclude) {
        const server_tokens a = cached.tokens.get_tokens_exclude_think(ctx, think);
        const server_tokens b = prompt.tokens.get_tokens_exclude_think(ctx, think);
        return a.size() > 0 && float(a.get_common_prefix(ctx, b).first) / a.size() >= 0.5f;
    }
    return float(cached.tokens.get_common_prefix(ctx, prompt.tokens).first) / cached.tokens.size() >= 0.5f;
}

bool server_prompt_cache::save_to_store(const server_prompt & prompt, size_t state_size, const std::function<bool(uint8_t *)> & fill,
                                        int id_slot, llama_pos pos_max) {
    // As alloc() does: nothing to do if the prompt is in the cache already, and the cached prompts that it
    // contains are older states of it. Here they are removed once the new file is written, not before.
    std::vector<std::list<server_prompt>::iterator> older;
    {
        auto tokens_ctx_shift = prompt.tokens.clone();
        tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
        for (auto it = states.begin(); it != states.end(); ++it) {
            const auto prefix = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
            if (prefix.second == tokens_ctx_shift.size()) {
                LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
                return false;
            }
            if (prefix.first == it->tokens.size()) {
                older.push_back(it);
            }
        }
    }
    // the file this slot took the conversation from is an older state of it too, whatever its thinking tokens
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->held_by == id_slot && std::find(older.begin(), older.end(), it) == older.end()) {
            if (starts_alike(*it, prompt)) {
                older.push_back(it);
            } else {
                it->held_by = -1; // the slot went on to another conversation
            }
        }
    }

    // the checkpoints whose bytes are here; one that was put on a disk of its own stays with the slot
    std::vector<server_prompt_store::ckpt_in> ckpts;
    for (const auto & c : prompt.checkpoints) {
        if (!c.data.empty()) {
            ckpts.push_back({ c.pos_min, c.pos_max, c.n_tokens, c.prompt_end, c.data.data(), c.data.size() });
        }
    }
    json meta = prompt.to_json();
    meta["think"] = json{
        { "exclude", prompt.think_tokens.exclude },
        { "begin",   prompt.think_tokens.begin },
        { "end",     prompt.think_tokens.end },
    };
    meta["pos_max"]    = pos_max;
    meta["n_ctx"]      = store_settings.n_ctx;
    meta["n_ubatch"]   = store_settings.n_ubatch;
    meta["n_parallel"] = store_settings.n_parallel;
    const std::string meta_text = meta.dump();

    const int64_t t_start = ggml_time_us();
    server_prompt_store::entry e;
    while (!store->write(meta_text, state_size, fill, ckpts, e)) {
        // No room in the directory: the older states of this conversation go first, then the oldest prompt, and
        // the file is tried again. Anything else: this prompt is not kept, the cache is as it was.
        if (!store->no_room || states.empty()) {
            LLAMA_LOG_WARN(" - the prompt is not kept: %s\n", store->error.c_str());
            return false;
        }
        auto victim = older.empty() ? states.begin() : older.back();
        LLAMA_LOG_INFO(" - %s: removing %s prompt of %d tokens (%.3f MiB)\n", store->error.c_str(),
            older.empty() ? "the oldest" : "an older state of this", victim->n_tokens(), victim->size() / (1024.0 * 1024.0));
        older.erase(std::remove(older.begin(), older.end(), victim), older.end());
        erase(victim);
    }
    for (auto it : older) {
        LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", it->n_tokens());
        erase(it);
    }
    e.meta.clear();
    auto & cur = states.emplace_back();
    cur.tokens             = prompt.tokens.clone();
    cur.n_kept_prompt      = prompt.n_kept_prompt;
    cur.n_discarded_prompt = prompt.n_discarded_prompt;
    cur.think_tokens       = prompt.think_tokens;
    cur.file               = std::move(e);
    cur.file_pos_max       = pos_max;
    LLAMA_LOG_INFO(" - written to %s with %zu checkpoints, %.3f MiB in %.2f ms\n", cur.file.path.c_str(), ckpts.size(),
        cur.size() / (1024.0 * 1024.0), (ggml_time_us() - t_start) / 1000.0);
    return true;
}

server_prompt* server_prompt_cache::alloc(const server_prompt& prompt, size_t state_size) {
    if (!forget_contained(prompt)) {
        return nullptr;
    }

    std::vector<uint8_t> state_data;

    // check if we can allocate enough memory for the new state
    try {
        state_data.resize(state_size);
    }
    catch (const std::bad_alloc& e) {
        LLAMA_LOG_INFO("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4 * size());

        LLAMA_LOG_INFO(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    // TODO: for some reason we can't copy server_tokens, so we have to do this workaround
    auto& cur = states.emplace_back();
    cur = {
        /*.tokens          =*/ prompt.tokens.clone(),
        /*.n_keep          =*/ prompt.n_kept_prompt,
        /*.n_discarded_prompt     =*/ prompt.n_discarded_prompt,
        /*.think_tokens                   =*/ prompt.think_tokens,
        /*.data            =*/ std::move(state_data),
        /*.checkpoints     =*/ prompt.checkpoints,
    };

    return &cur;
}


void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            LLAMA_LOG_INFO(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            erase(states.begin());
        }

        // Trim checkpoints when the remaining state exceeds the limit. Keep the
        // prompt-end checkpoint while another checkpoint can be evicted instead.
        for (auto& state : states) {
            while (size() > limit_size && !state.checkpoints.empty()) {
                auto it = state.checkpoints.begin();
                if (it->prompt_end && std::next(it) != state.checkpoints.end()) {
                    ++it;
                }
                LLAMA_LOG_INFO(" - cache size limit reached, dropping a checkpoint (size = %.3f MiB)\n",
                    it->size() / (1024.0 * 1024.0));
                if (!it->spill_path.empty()) {
                    remove(it->spill_path.c_str());
                }
                state.checkpoints.erase(it);
            }
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size / size_per_token) : limit_tokens;

    LLAMA_LOG_INFO(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
        states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto& state : states) {
        LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, %9.3f MiB\n",
            (const void*)&state, state.n_tokens(), state.n_discarded_prompt,
            state.in_file() ? (size_t)state.file.n_ckpt : state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
