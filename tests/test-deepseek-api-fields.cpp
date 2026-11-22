// What a client written for DeepSeek's API sends and reads back, with the DeepSeek-V4.1 template:
//   - "thinking": {"type": ...} and "reasoning_effort" of a request reach the template;
//   - the line breaks the model writes before its block of calls are not part of the text of the answer, so a
//     client that sends the answer back gets the prompt the model wrote.

#include "chat.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        n_fail++;
    }
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool contains(const std::string & text, const std::string & part) {
    return text.find(part) != std::string::npos;
}

static bool ends_with(const std::string & text, const std::string & end) {
    return text.size() >= end.size() && text.compare(text.size() - end.size(), end.size(), end) == 0;
}

// the arguments of the template a request gives, the way the server reads them
static std::map<std::string, std::string> kwargs_of(const std::string & body) {
    std::map<std::string, std::string> kwargs;
    common_chat_reasoning_kwargs_parse_oaicompat(json::parse(body), kwargs);
    return kwargs;
}

static common_chat_params render(const common_chat_templates * tmpls, const std::string & body, bool with_tool = false) {
    common_chat_msg user;
    user.role    = "user";
    user.content = "Hello";

    common_chat_templates_inputs inputs;
    inputs.messages             = { user };
    inputs.chat_template_kwargs = kwargs_of(body);
    inputs.reasoning_format     = COMMON_REASONING_FORMAT_DEEPSEEK;
    // the server takes `enable_thinking` out of the arguments
    const auto it = inputs.chat_template_kwargs.find("enable_thinking");
    if (it != inputs.chat_template_kwargs.end()) {
        inputs.enable_thinking = it->second == "true";
    }
    if (with_tool) {
        inputs.tools = { common_chat_tool{
            /* .name        = */ "get_goal",
            /* .description = */ "Read the goal",
            /* .parameters  = */ R"({"type":"object","properties":{"detail":{"type":"string"}}})",
        } };
    }
    return common_chat_templates_apply(tmpls, inputs);
}

static common_chat_msg parse(const common_chat_params & params, const std::string & output) {
    common_peg_arena arena;
    arena.load(params.parser);
    common_chat_parser_params parser_params(params);
    parser_params.parser           = arena;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    return common_chat_parse(output, /* is_partial = */ false, parser_params);
}

int main() {
    // --- the fields of the request, as arguments of the template ---------------------------------------------
    {
        auto k = kwargs_of(R"({"thinking": {"type": "disabled"}})");
        check(k.size() == 1 && k["enable_thinking"] == "false", "thinking disabled");
        k = kwargs_of(R"({"thinking": {"type": "enabled"}, "reasoning_effort": "max"})");
        check(k.size() == 2 && k["enable_thinking"] == "true" && k["reasoning_effort"] == "\"max\"", "thinking enabled, effort max");
        k = kwargs_of(R"({"reasoning_effort": 40})");
        check(k.size() == 1 && k["reasoning_effort"] == "40", "effort as a number");
        k = kwargs_of(R"({"reasoning_effort": "none"})");
        check(k.size() == 1 && k["enable_thinking"] == "false", "effort none turns the reasoning off");
        // a budget of Anthropic's API, another type, no field at all: nothing changes
        k = kwargs_of(R"({"thinking": {"type": "adaptive"}, "reasoning_effort": null, "messages": []})");
        check(k.empty(), "fields that say nothing");
        // what the command line gave stays where the request says nothing
        k = { { "enable_thinking", "false" }, { "reasoning_effort", "\"low\"" } };
        common_chat_reasoning_kwargs_parse_oaicompat(json::parse(R"({"reasoning_effort": "high"})"), k);
        check(k["enable_thinking"] == "false" && k["reasoning_effort"] == "\"high\"", "the request over the command line");
    }

    const std::string src = read_file("models/templates/deepseek-ai-DeepSeek-V4.1.jinja");
    common_chat_templates_ptr tmpls(common_chat_templates_init(/* model = */ nullptr, src));

    // --- the prompt ------------------------------------------------------------------------------------------
    {
        auto p = render(tmpls.get(), R"({})");
        check(ends_with(p.prompt, "<｜Assistant｜><think>") && contains(p.prompt, "Reasoning Effort: 75 "), "defaults: thinking, effort 75");
        p = render(tmpls.get(), R"({"thinking": {"type": "disabled"}})");
        check(ends_with(p.prompt, "<｜Assistant｜></think>") && !contains(p.prompt, "Reasoning Effort"), "thinking disabled in the prompt");
        p = render(tmpls.get(), R"({"reasoning_effort": "none"})");
        check(ends_with(p.prompt, "<｜Assistant｜></think>"), "effort none in the prompt");

        const std::pair<const char *, const char *> levels[] = {
            { R"("low")", "50" }, { R"("high")", "75" }, { R"("max")", "100" }, { "33", "33" },
            // the levels of the other APIs
            { R"("minimal")", "50" }, { R"("medium")", "75" }, { R"("xhigh")", "100" },
        };
        for (const auto & [effort, budget] : levels) {
            p = render(tmpls.get(), std::string(R"({"thinking": {"type": "enabled"}, "reasoning_effort": )") + effort + "}");
            check(contains(p.prompt, std::string("Reasoning Effort: ") + budget + " ") && ends_with(p.prompt, "<think>"),
                  std::string("effort ") + effort + " is " + budget);
        }

        // a level nobody knows does not make the request fail: the default amount
        for (const char * effort : { R"("banana")", "0", "250", "1.5" }) {
            p = render(tmpls.get(), std::string(R"({"reasoning_effort": )") + effort + "}");
            check(contains(p.prompt, "Reasoning Effort: 75 "), std::string("effort ") + effort + " is the default");
        }
    }

    // --- the answer ------------------------------------------------------------------------------------------
    {
        const std::string calls =
            "<｜DSML｜ calls>\n"
            "<｜DSML｜ invoke name=\"get_goal\">\n"
            "<｜DSML｜ parameter name=\"detail\" string=\"true\">all</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls>";
        const auto with_thinking = render(tmpls.get(), R"({})", /* with_tool = */ true);

        auto msg = parse(with_thinking, "I look at the goal.</think>\n\n" + calls);
        check(msg.reasoning_content == "I look at the goal.", "reasoning before a call");
        check(msg.content.empty(), "no text before the call, not even its line breaks: '" + msg.content + "'");
        check(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "get_goal" &&
              json::parse(msg.tool_calls[0].arguments) == json::parse(R"({"detail": "all"})"), "the call");

        msg = parse(with_thinking, "Hm.</think>Let me look.\n\n" + calls);
        check(msg.content == "Let me look." && msg.tool_calls.size() == 1, "text, then a call: '" + msg.content + "'");

        msg = parse(with_thinking, "Hm.</think>" + calls);
        check(msg.content.empty() && msg.tool_calls.size() == 1, "a call right after the reasoning");

        // an answer without a call keeps its text, the line breaks inside it too; those at its very end go
        msg = parse(with_thinking, "Hm.</think>Hello.\n\nHow are you?\n");
        check(msg.content == "Hello.\n\nHow are you?" && msg.tool_calls.empty(), "an answer without a call: '" + msg.content + "'");
        msg = parse(with_thinking, "Hm.</think>a < b, and <｜DS is no call");
        check(msg.content == "a < b, and <｜DS is no call" && msg.tool_calls.empty(), "text that looks like a call: '" + msg.content + "'");

        const auto without_thinking = render(tmpls.get(), R"({"thinking": {"type": "disabled"}})", /* with_tool = */ true);
        msg = parse(without_thinking, "\n\n" + calls);
        check(msg.content.empty() && msg.reasoning_content.empty() && msg.tool_calls.size() == 1, "a call with the thinking off");
    }

    if (n_fail == 0) {
        printf("the request fields and the answers of DeepSeek's API are handled\n");
    }
    return n_fail ? 1 : 0;
}
