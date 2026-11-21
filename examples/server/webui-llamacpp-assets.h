// The llama.cpp web UI ("Llama Chat") as it is built upstream: many files, not one page.
// scripts/webui-assets.cmake writes the table from examples/server/public_llamacpp.

#pragma once

#include <cstddef>

struct webui_asset {
    const char          * name;  // path below the root, without the leading slash
    const unsigned char * data;
    size_t                size;
    const char          * etag;  // quoted, for ETag and If-None-Match
    const char          * type;  // MIME type
};

extern const webui_asset webui_llamacpp_assets[];
extern const size_t      webui_llamacpp_n_assets;
// the data is gzip-compressed and goes out with Content-Encoding: gzip
extern const bool        webui_llamacpp_gzip;
