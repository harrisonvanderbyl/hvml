#ifndef TOKENIZERS_QWEN2_HPP
#define TOKENIZERS_QWEN2_HPP

//
//  Qwen2 byte-level BPE tokenizer (Qwen2 / Qwen2.5 / Qwen3 / Qwen3-ASR).
//
//  load(dir) reads vocab.json + merges.txt (or tokenizer.json) and the added
//  tokens from tokenizer_config.json / tokenizer.json.  encode() follows the
//  Qwen2 pre-tokenizer regex
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//     | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//  (with compact tables for the Unicode letter / number / mark classes), then BPE.
//

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <climits>
#include "file_loaders/json.hpp"

struct Qwen2Tokenizer {
    std::unordered_map<std::string, int> vocab;          // byte-mapped token → id
    std::vector<std::string> id_to_token;                 // id → byte-mapped token (or added token text)
    std::unordered_map<std::string, int> merge_rank;      // "a b" → rank
    std::map<std::string, int> added;                     // added/special token text → id
    std::vector<bool> is_added;
    bool loaded = false;

    // ---- byte ↔ unicode (GPT-2) ------------------------------------------

    std::string byte_to_str[256];
    std::unordered_map<std::string, unsigned char> str_to_byte;

    static std::string utf8(uint32_t cp) {
        std::string s;
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    // Decode one code point at s[i]; returns its byte length.
    static int next_cp(const std::string& s, size_t i, uint32_t& cp) {
        unsigned char c = (unsigned char)s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + len > s.size()) len = 1;
        if (len == 1) { cp = c; return 1; }
        cp = c & (0xFF >> (len + 1));
        for (int k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        return len;
    }

    Qwen2Tokenizer() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; b++) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            byte_to_str[bs[i]] = utf8((uint32_t)cs[i]);
            str_to_byte[byte_to_str[bs[i]]] = (unsigned char)bs[i];
        }
    }

    // ---- loading -------------------------------------------------------------

    void set_token(int id, const std::string& tok, bool added_token) {
        if ((int)id_to_token.size() <= id) {
            id_to_token.resize(id + 1);
            is_added.resize(id + 1, false);
        }
        id_to_token[id] = tok;
        is_added[id] = added_token;
    }

    bool load(const std::string& dir) {
        using json = nlohmann::json;
        std::ifstream vf(dir + "/vocab.json"), mf(dir + "/merges.txt"), tf(dir + "/tokenizer.json");
        if (vf && mf) {
            json v = json::parse(vf);
            for (auto& [tok, id] : v.items()) {
                vocab[tok] = id.get<int>();
                set_token(id.get<int>(), tok, false);
            }
            std::string line;
            int rank = 0;
            while (std::getline(mf, line)) {
                if (line.empty() || line.rfind("#version", 0) == 0) continue;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                merge_rank[line] = rank++;
            }
        } else if (tf) {
            json t = json::parse(tf);
            for (auto& [tok, id] : t["model"]["vocab"].items()) {
                vocab[tok] = id.get<int>();
                set_token(id.get<int>(), tok, false);
            }
            int rank = 0;
            for (auto& m : t["model"]["merges"]) {
                std::string key = m.is_string() ? m.get<std::string>()
                                                : m[0].get<std::string>() + " " + m[1].get<std::string>();
                merge_rank[key] = rank++;
            }
            for (auto& a : t["added_tokens"]) add_token(a["content"].get<std::string>(), a["id"].get<int>());
        } else {
            return false;
        }

        std::ifstream cf(dir + "/tokenizer_config.json");
        if (cf) {
            json c = json::parse(cf);
            if (c.contains("added_tokens_decoder")) {
                for (auto& [id, info] : c["added_tokens_decoder"].items()) {
                    add_token(info["content"].get<std::string>(), std::stoi(id));
                }
            }
        }
        loaded = true;
        return true;
    }

    void add_token(const std::string& text, int id) {
        added[text] = id;
        set_token(id, text, true);
    }

    // ---- decoding --------------------------------------------------------------

    std::string decode(const std::vector<int>& ids, bool skip_special = false) const {
        std::string out;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token.size() || id_to_token[id].empty()) {
                out += "<" + std::to_string(id) + ">";
                continue;
            }
            if (is_added[id]) {
                if (!skip_special) out += id_to_token[id];
                continue;
            }
            const std::string& tok = id_to_token[id];
            for (size_t i = 0; i < tok.size();) {
                uint32_t cp;
                int len = next_cp(tok, i, cp);
                auto it = str_to_byte.find(tok.substr(i, len));
                if (it != str_to_byte.end()) out += (char)it->second;
                else out += tok.substr(i, len);
                i += len;
            }
        }
        return out;
    }

    // ---- pre-tokenisation --------------------------------------------------------

    static bool is_space(uint32_t c) {
        return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
               (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }
    // \p{N}: decimal digits plus other numeric characters (superscripts,
    // vulgar fractions, number forms, circled numbers).
    static bool is_digit(uint32_t c) {
        return (c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19) || (c >= 0x0660 && c <= 0x0669) ||
               (c >= 0x06F0 && c <= 0x06F9) || (c >= 0x0966 && c <= 0x096F) ||
               c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
               c == 0x2070 || (c >= 0x2074 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) ||
               (c >= 0x2150 && c <= 0x2189) || (c >= 0x2460 && c <= 0x249B) || (c >= 0x24EA && c <= 0x24FF) ||
               (c >= 0x2776 && c <= 0x2793) || (c >= 0x3021 && c <= 0x3029) || (c >= 0x3007 && c <= 0x3007);
    }
    // \p{M}: combining marks (accents, Indic vowel signs and viramas, Arabic
    // harakat, Thai vowels/tones, variation selectors...).  Not letters.
    static bool is_mark(uint32_t c) {
        if (c >= 0x0300 && c <= 0x036F) return true;
        if (c >= 0x0483 && c <= 0x0489) return true;
        if (c >= 0x0591 && c <= 0x05C7 && c != 0x05BE && c != 0x05C0 && c != 0x05C3 && c != 0x05C6) return true;
        if ((c >= 0x0610 && c <= 0x061A) || (c >= 0x064B && c <= 0x065F) || c == 0x0670 ||
            (c >= 0x06D6 && c <= 0x06DC) || (c >= 0x06DF && c <= 0x06E4) || (c >= 0x06E7 && c <= 0x06E8) ||
            (c >= 0x06EA && c <= 0x06ED)) return true;
        if (c >= 0x0900 && c <= 0x0DFF) {               // Indic blocks share one layout
            uint32_t o = c & 0x7F;
            return o <= 0x03 || o == 0x3C || (o >= 0x3E && o <= 0x4F) || (o >= 0x51 && o <= 0x57) ||
                   (o >= 0x62 && o <= 0x63);
        }
        if (c == 0x0E31 || (c >= 0x0E34 && c <= 0x0E3A) || (c >= 0x0E47 && c <= 0x0E4E)) return true;   // Thai
        if (c == 0x0EB1 || (c >= 0x0EB4 && c <= 0x0EBC) || (c >= 0x0EC8 && c <= 0x0ECD)) return true;   // Lao
        if ((c >= 0x0F18 && c <= 0x0F19) || (c >= 0x0F71 && c <= 0x0F84) || (c >= 0x0F8D && c <= 0x0FBC)) return true;
        if (c >= 0x102B && c <= 0x103E) return true;                                                     // Myanmar
        if ((c >= 0x1AB0 && c <= 0x1AFF) || (c >= 0x1DC0 && c <= 0x1DFF) || (c >= 0x20D0 && c <= 0x20FF)) return true;
        if ((c >= 0x302A && c <= 0x302F) || (c >= 0x3099 && c <= 0x309A)) return true;
        if ((c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F)) return true;
        if (c >= 0xE0100 && c <= 0xE01EF) return true;
        return false;
    }

    static bool is_letter(uint32_t c) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
        if (c < 0x80) return false;
        if (is_space(c) || is_digit(c) || is_mark(c)) return false;
        if (c <= 0xBF) return c == 0xAA || c == 0xB5 || c == 0xBA;          // Latin-1 punctuation/symbols
        if (c == 0xD7 || c == 0xF7) return false;
        if (c >= 0x2000 && c <= 0x2BFF) return false;                         // punctuation, arrows, maths, symbols
        if (c >= 0x3000 && c <= 0x303F) return false;                         // CJK punctuation
        if (c >= 0xFE30 && c <= 0xFE4F) return false;
        if ((c >= 0xFF01 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) ||
            (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) return false;
        if (c >= 0x1F000) return false;                                       // emoji and pictographs
        return true;
    }

    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<uint32_t> cps;
        std::vector<size_t> offs;
        for (size_t i = 0; i < text.size();) {
            uint32_t cp;
            int len = next_cp(text, i, cp);
            cps.push_back(cp);
            offs.push_back(i);
            i += len;
        }
        offs.push_back(text.size());
        size_t n = cps.size();
        auto L = [&](size_t i) { return i < n && is_letter(cps[i]); };
        auto N = [&](size_t i) { return i < n && is_digit(cps[i]); };
        auto S = [&](size_t i) { return i < n && is_space(cps[i]); };
        auto NL = [&](size_t i) { return i < n && (cps[i] == '\r' || cps[i] == '\n'); };

        std::vector<std::string> out;
        size_t i = 0;
        while (i < n) {
            size_t j = i;
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (cps[i] == '\'' && i + 1 < n) {
                auto lower = [&](size_t k) { return k < n ? (uint32_t)std::tolower((int)(cps[k] < 128 ? cps[k] : 0)) : 0u; };
                uint32_t a = lower(i + 1), b = lower(i + 2);
                if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
                else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
            }
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (j == i) {
                size_t k = i;
                if (!L(k) && !N(k) && !NL(k) && L(k + 1)) k++;
                if (L(k)) {
                    while (L(k)) k++;
                    j = k;
                }
            }
            // \p{N}
            if (j == i && N(i)) j = i + 1;
            // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            if (j == i) {
                size_t k = i;
                if (cps[k] == ' ') k++;
                if (k < n && !S(k) && !L(k) && !N(k)) {
                    while (k < n && !S(k) && !L(k) && !N(k)) k++;
                    while (NL(k)) k++;
                    j = k;
                }
            }
            // \s*[\r\n]+
            if (j == i && S(i)) {
                size_t k = i, last_nl = SIZE_MAX;
                while (S(k)) {
                    if (NL(k)) last_nl = k;
                    k++;
                }
                if (last_nl != SIZE_MAX) j = last_nl + 1;
                // \s+(?!\S)  then  \s+
                else if (k == n || k - i == 1) j = k;
                else j = k - 1;
            }
            if (j == i) j = i + 1;   // anything else: single character
            out.push_back(text.substr(offs[i], offs[j] - offs[i]));
            i = j;
        }
        return out;
    }

    // ---- BPE -------------------------------------------------------------------

    void bpe(const std::string& piece, std::vector<int>& ids) const {
        std::vector<std::string> sym;
        for (unsigned char b : piece) sym.push_back(byte_to_str[b]);
        while (sym.size() > 1) {
            int best = INT_MAX;
            size_t at = 0;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                auto it = merge_rank.find(sym[i] + " " + sym[i + 1]);
                if (it != merge_rank.end() && it->second < best) { best = it->second; at = i; }
            }
            if (best == INT_MAX) break;
            std::string a = sym[at], b = sym[at + 1];
            std::vector<std::string> next;
            for (size_t i = 0; i < sym.size();) {
                if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) { next.push_back(a + b); i += 2; }
                else next.push_back(sym[i++]);
            }
            sym.swap(next);
        }
        for (auto& s : sym) {
            auto it = vocab.find(s);
            if (it != vocab.end()) ids.push_back(it->second);
            else for (unsigned char c : s) ids.push_back(vocab.at(byte_to_str[c]));   // unreachable for a byte-level vocab
        }
    }

    std::vector<int> encode(const std::string& text) const {
        if (!loaded) throw std::runtime_error("Qwen2Tokenizer: no tokenizer loaded (vocab.json/merges.txt or tokenizer.json)");
        std::vector<int> ids;
        size_t i = 0, start = 0;
        auto flush = [&](size_t end) {
            if (end > start) for (auto& p : pretokenize(text.substr(start, end - start))) bpe(p, ids);
        };
        while (i < text.size()) {
            // longest added token starting here
            int id = -1;
            size_t len = 0;
            if (text[i] == '<') {
                for (auto& [tok, tid] : added) {
                    if (tok.size() > len && text.compare(i, tok.size(), tok) == 0) { id = tid; len = tok.size(); }
                }
            }
            if (id >= 0) {
                flush(i);
                ids.push_back(id);
                i += len;
                start = i;
            } else {
                i++;
            }
        }
        flush(text.size());
        return ids;
    }
};

#endif // TOKENIZERS_QWEN2_HPP
