/*
 * Copyright (c) 2025 mocibb (mocibb@163.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#define USE_PARALLEL_HASH

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#ifdef USE_PARALLEL_HASH
#include <parallel_hashmap/phmap.h>
#endif
#include "LRUCache11.hpp"
#include <limits>
#include <chrono>
#include <iostream>

using Bytes = std::vector<unsigned char>;
using PairBytes = std::pair<Bytes, Bytes>;

#ifdef USE_PARALLEL_HASH
#define HashSet phmap::node_hash_set
#define HashMap phmap::node_hash_map
#else
#define HashSet std::unordered_set
#define HashMap std::unordered_map
#endif

namespace std {

    template<>
    struct hash<Bytes> {
        size_t operator()(const Bytes& vec) const noexcept {
            auto data = reinterpret_cast<const char*>(vec.data());
            return hash<string_view>{}(string_view(data, vec.size()));
        }
    };
}

namespace {

inline Bytes to_bytes(const Bytes& first, const Bytes& second) {
    Bytes bytes;
    bytes.reserve(first.size() + second.size());

    bytes.insert(bytes.end(), first.begin(),  first.end());
    bytes.insert(bytes.end(), second.begin(), second.end());

    return bytes;
}

class Encoder {
public:
    Encoder(std::unordered_map<Bytes, int>&& vocab2ids, int cache_size=10000);
    std::vector<int> _encode(const std::string& text);
private:
    HashMap<Bytes, int> vocab2ids_;
    lru11::Cache<std::string, std::vector<int>> cache_;
};

Encoder::Encoder(std::unordered_map<Bytes, int>&& vocab2ids, int cache_size) : cache_(cache_size) {
    vocab2ids_.reserve(vocab2ids.size());
   
    for (auto&& [bytes, v] : vocab2ids) {
        vocab2ids_.emplace(std::move(bytes), v);
    }
}

std::vector<int> Encoder::_encode(const std::string& text) {
    if (cache_.contains(text)) {
        return cache_.get(text);
    }

    std::vector<Bytes> word;
    word.reserve(text.size());
    for (const auto b : text) {
        word.emplace_back(Bytes{static_cast<unsigned char>(b)});
    }

    while(true) {
        int best_value = std::numeric_limits<int>::max();
        size_t best_idx = std::numeric_limits<size_t>::max();
        Bytes best_bytes;
        for (size_t i = 0; i < word.size()-1; i++) {
            Bytes pair_bytes = to_bytes(word[i], word[i+1]);
            if (auto it = vocab2ids_.find(pair_bytes); it != vocab2ids_.end()) {
                if (it->second < best_value) {
                    best_value = it->second;
                    best_idx = i;
                    best_bytes = pair_bytes;
                }
            }
        }
        if (best_bytes.empty()) {
            break;
        }
        
        auto it = word.begin() + best_idx;
        it = word.erase(it, it+2);
        word.insert(it, best_bytes);
    }

    std::vector<int> ids;
    ids.reserve(word.size());

    for (const auto& w : word) {
        ids.emplace_back(vocab2ids_[w]);
    }

    cache_.insert(text, ids);

    return ids;
}

} //end namespace
