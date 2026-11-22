// The grammar that holds the arguments of a DeepSeek-V4.1 tool call to the schema of the tool, for a tool declared
// the way Metabase declares its query builder: arrays whose items are `{}`, that is anything, and a stage object
// that does not list all the properties its own instructions name (`limit`).
//
// `{}` used to be taken for "a JSON object", so the grammar let nothing but objects into these arrays: a model
// that means three strings for `source-table` cannot write them, and calls the tool again and again.

#include "chat.h"
#include "../src/llama-grammar.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

static int n_fail = 0;

static std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// whether the grammar takes the whole text and is complete at its end
static bool accepts(const std::string & grammar_str, const std::string & text) {
    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (grammar == nullptr) {
        throw std::runtime_error("the grammar does not parse:\n" + grammar_str);
    }
    bool ok = true;
    try {
        llama_grammar_accept_str(*grammar, text);
    } catch (const std::runtime_error &) {
        ok = false;
    }
    if (ok) {
        ok = false;
        for (const auto & stack : llama_grammar_get_stacks(grammar)) {
            ok = ok || stack.empty();
        }
    }
    llama_grammar_free_impl(grammar);
    return ok;
}

static void expect(const std::string & what, const std::string & grammar, const std::string & text, bool accepted) {
    const bool got = accepts(grammar, text);
    if (got != accepted) {
        fprintf(stderr, "FAILED: %s: %s, and should be %s\n  %s\n", what.c_str(), got ? "accepted" : "rejected",
                accepted ? "accepted" : "rejected", text.c_str());
        n_fail++;
    }
}

// The `query` argument as Metabase describes it (src/metabase/metabot/tools/construct.clj), without the prose.
static const char * k_query_schema = R"({
    "type": "object",
    "required": ["lib/type", "stages"],
    "properties": {
        "lib/type": {"type": "string", "const": "mbql/query"},
        "stages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "lib/type":     {"type": "string", "const": "mbql.stage/mbql"},
                    "source-table": {"type": "array", "minItems": 3, "maxItems": 3, "items": {}},
                    "source-card":  {"type": "string"},
                    "aggregation":  {"type": "array", "items": {"type": "array", "items": {}}},
                    "breakout":     {"type": "array", "items": {"type": "array", "items": {}}},
                    "filters":      {"type": "array", "items": {"type": "array", "items": {}}},
                    "fields":       {"type": "array", "items": {"type": "array", "items": {}}},
                    "order-by":     {"type": "array", "items": {"type": "array", "items": {}}},
                    "joins":        {"type": "array", "items": {"type": "object"}},
                    "expressions":  {"type": "object"}
                }
            }
        }
    }
})";

static std::string call(const std::string & query) {
    return "<｜DSML｜ calls>\n"
           "<｜DSML｜ invoke name=\"construct_notebook_query\">\n"
           "<｜DSML｜ parameter name=\"query\" string=\"false\">" + query + "</｜DSML｜ parameter>\n"
           "<｜DSML｜ parameter name=\"title\" string=\"true\">Orders this month</｜DSML｜ parameter>\n"
           "</｜DSML｜ invoke>\n"
           "</｜DSML｜ calls>";
}

int main() {
    // DeepSeek's own template, and the port of it that Unsloth puts in its GGUF files of DeepSeek-V4.1: the one
    // a server started without --chat-template-file works with
    for (const char * path : { "models/templates/deepseek-ai-DeepSeek-V4.1.jinja",
                               "models/templates/unsloth-DeepSeek-V4.1-Flash-1017.jinja" }) {
    const int n_fail_before = n_fail;
    const std::string src = read_file(path);
    common_chat_templates_ptr tmpls(common_chat_templates_init(/* model = */ nullptr, src));

    common_chat_msg user;
    user.role    = "user";
    user.content = "How many orders this month?";

    // the example in the description of the tool: strings and null in `source-table`, a clause `["count", {}]`
    const std::string example =
        R"({"lib/type": "mbql/query", "stages": [{"lib/type": "mbql.stage/mbql", )"
        R"("source-table": ["Sample Database", null, "ORDERS"], "aggregation": [["count", {}]], )"
        R"("breakout": [["field", {"temporal-unit": "month"}, ["Sample Database", null, "ORDERS", "CREATED_AT"]]]}]})";
    // `limit`, which the instructions of the tool name and its schema does not, and the properties in another order
    const std::string with_limit =
        R"({"stages": [{"source-table": ["Sample Database", "PUBLIC", "ORDERS"], "limit": 10, )"
        R"("order-by": [["desc", {}, ["field", {}, ["Sample Database", "PUBLIC", "ORDERS", "CREATED_AT"]]]], )"
        R"("aggregation": [["count", {}]], "lib/type": "mbql.stage/mbql"}], "lib/type": "mbql/query"})";
    // what the grammar used to force: objects where the strings belong
    const std::string objects =
        R"({"lib/type": "mbql/query", "stages": [{"lib/type": "mbql.stage/mbql", )"
        R"("source-table": [{"db": "Sample Database"}, {"schema": "PUBLIC"}, {"table": "ORDERS"}]}]})";
    const std::string not_an_array =
        R"({"lib/type": "mbql/query", "stages": [{"lib/type": "mbql.stage/mbql", "source-table": {"table": "ORDERS"}}]})";
    const std::string two_items =
        R"({"lib/type": "mbql/query", "stages": [{"lib/type": "mbql.stage/mbql", "source-table": ["Sample Database", "ORDERS"]}]})";
    const std::string wrong_const =
        R"({"lib/type": "query", "stages": [{"source-table": ["Sample Database", "PUBLIC", "ORDERS"]}]})";

    for (const bool strict : { false, true }) {
        common_chat_tool tool{
            /* .name        = */ "construct_notebook_query",
            /* .description = */ "Build a query",
            /* .parameters  = */ std::string(R"({"type":"object","properties":{"query":)") + k_query_schema +
                                 R"(,"title":{"type":"string"}},"required":["query","title"]})",
            /* .strict      = */ strict,
        };
        common_chat_templates_inputs inputs;
        inputs.messages = { user };
        inputs.tools    = { tool };
        const auto params = common_chat_templates_apply(tmpls.get(), inputs);
        if (params.grammar.empty() || !params.grammar_lazy) {
            fprintf(stderr, "FAILED: no lazy grammar for the tool call\n");
            return 1;
        }
        const std::string mode = strict ? "strict tool" : "tool";

        // anything goes into an array of `{}`
        expect(mode + ", the example of the tool", params.grammar, call(example), true);
        // the schema still says what it says
        expect(mode + ", source-table as an object", params.grammar, call(not_an_array), false);
        expect(mode + ", source-table of two items", params.grammar, call(two_items), false);
        expect(mode + ", another lib/type",          params.grammar, call(wrong_const), false);
        // objects in the array are JSON values like any other
        expect(mode + ", objects in source-table",   params.grammar, call(objects), true);
        // a property the schema does not name, and another order: for the tool that is not strict only
        expect(mode + ", limit and another order",   params.grammar, call(with_limit), !strict);
    }

    if (n_fail != n_fail_before) {
        fprintf(stderr, "  with %s\n", path);
    }
    }

    if (n_fail == 0) {
        printf("the arguments of the tool call are held as the schema says\n");
    }
    return n_fail ? 1 : 0;
}
