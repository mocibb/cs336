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
#include <vector>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <memory>
#include <chrono>
#include <iostream>

using Bytes = std::vector<unsigned char>;
using PairBytes = std::pair<Bytes, Bytes>;

namespace std {

    template<>
    struct hash<Bytes> {
        size_t operator()(const Bytes& vec) const noexcept {
            auto data = reinterpret_cast<const char*>(vec.data());
            return hash<string_view>{}(string_view(data, vec.size()));
        }
    };

    template<>
    struct hash<PairBytes> {
        size_t operator()(const PairBytes& p) const noexcept {
            return hash<Bytes>{}(p.first) ^ (hash<Bytes>{}(p.second) << 1);
        }
    };
}

namespace {

template<class T> class LazyOrderedQueue;
using Queue = LazyOrderedQueue<PairBytes>;

Bytes to_bytes(std::string_view sv) {
    return Bytes(sv.begin(), sv.end());
}

Bytes pair_bytes(const PairBytes& pair) {
    Bytes bytes;
    bytes.reserve(pair.first.size() + pair.second.size());

    bytes.insert(bytes.end(), pair.first.begin(),  pair.first.end());
    bytes.insert(bytes.end(), pair.second.begin(), pair.second.end());

    return bytes;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Bytes>& bArray) {
    for (auto bytes : bArray) {
        os << std::string(bytes.begin(), bytes.end());
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const PairBytes& pair) {
    os << "(b'" << std::string(pair.first.begin(), pair.first.end());
    os << "', b'";
    os << std::string(pair.second.begin(), pair.second.end());
    return os << "')";
}


template <typename T>
struct MostFreqLexComparator {
    bool operator()(const std::pair<T, int>& a, const std::pair<T, int>& b) {
        return std::tie(a.second, a.first) < std::tie(b.second, b.first); 
    }
};

template<class T>
class LazyOrderedQueue {
public:
    LazyOrderedQueue() = default;

    LazyOrderedQueue(std::unordered_map<T, int>&& entry_map): entry_map_(std::move(entry_map)) {
        for (const auto& entry : entry_map_) {
            queue_.emplace(entry.first, entry.second);
        }
    }

    void emplace(T&& key, int value) {
        auto [it, _] = entry_map_.insert_or_assign(std::forward<T>(key), value);
        queue_.emplace(it->first, value);
    }

    void increase(T&& key, int delta) {
        auto v = this->operator[](key);
        emplace(std::forward<T>(key), v + delta);
    }    

    int operator[] (const T& key) {
        auto it = entry_map_.find(key);
        return it == entry_map_.end() ? 0 : entry_map_[key];
    }

    T pop() {
        while(!queue_.empty()) {
            auto top = queue_.top(); queue_.pop();

            auto it = entry_map_.find(top.first);
            if (it != entry_map_.end() && it->second == top.second) {
                T key = std::move(top.first);
                entry_map_.erase(it);
                return key;
           }
        }
        throw std::runtime_error("Queue is empty");
    }

    bool empty() const { return entry_map_.empty(); }
    size_t size() const { return entry_map_.size(); }

private :
    std::priority_queue<std::pair<T, int>, 
                        std::vector<std::pair<T, int>>, 
                        MostFreqLexComparator<T>> queue_;
    std::unordered_map<T, int> entry_map_;
};

class BytePairEncoding {
public:
    constexpr static int ASCII_SIZE = 256;
    BytePairEncoding(int vocab_size, const std::vector<std::string>& special_tokens);
    void setVocabFreq(std::unordered_map<Bytes, int>&& vocab_freqs);
    std::tuple< std::unordered_map<size_t, Bytes>, std::vector<PairBytes> > merge();
private:
    std::tuple<std::vector<PairBytes>, std::vector<Bytes>, int> 
    mergePair(size_t voc_idx, const PairBytes& best_pair);

    size_t vocab_size_;
    // 
    std::unordered_map<size_t, Bytes> vocab_;
    // 所有词汇表
    std::vector<std::vector<Bytes>> all_vocab_list_;
    // vocab和freq
    std::unordered_map<size_t, int> all_vocab_freq_map_;
    // pair和freq的堆
    std::unique_ptr<Queue> pairsfreq_queue_;
    // 从pair到vocab的映射
    std::unordered_map<PairBytes, std::unordered_set<size_t>> pairs2vocab_map_;
};


BytePairEncoding::BytePairEncoding(int vocab_size, 
                                   const std::vector<std::string>& special_tokens) :
    vocab_size_(vocab_size) {
    for(int k = 0; k < ASCII_SIZE; k++) {
        vocab_[k] = Bytes{static_cast<unsigned char>(k)};
    }

    for (size_t k = 0; k < special_tokens.size(); k++) {
        vocab_[ASCII_SIZE + k] = to_bytes(special_tokens[k]);  
    }
}

void BytePairEncoding::setVocabFreq(std::unordered_map<Bytes, int>&& vocab_freqs) {
    all_vocab_list_.reserve(vocab_freqs.size());
    all_vocab_freq_map_.reserve(vocab_freqs.size());
    for (auto&& [bytes, v] : vocab_freqs) {
        std::vector<Bytes> voc;
        voc.reserve(bytes.size());
        for (const auto b : bytes) {
            voc.emplace_back(1, b);
        }
        all_vocab_list_.emplace_back(std::move(voc));
        all_vocab_freq_map_[all_vocab_list_.size()-1] = v;
    }
}

std::tuple<std::unordered_map<size_t, Bytes>, std::vector<PairBytes> > 
    BytePairEncoding::merge() {
    // 1. 构建 pairsfreq_queue_ 和 pairs2freq_map_
    std::unordered_map<PairBytes, int> pairsfreq_map;
    for (size_t i = 0; i < all_vocab_list_.size(); i++) {
        const auto& voc = all_vocab_list_[i];
        const auto len = voc.size();
        const auto freq = all_vocab_freq_map_[i];
        for (size_t k = 0; k < len - 1; k ++) {
            const auto pair = std::make_pair(voc[k], voc[k+1]);
            if (pairsfreq_map.find(pair) == pairsfreq_map.end()) {
                pairsfreq_map.emplace(pair, freq);
            } else {
                pairsfreq_map.at(pair) += freq;
            }
            pairs2vocab_map_[pair].insert(i);
        }
    }

    pairsfreq_queue_.reset(new Queue(std::move(pairsfreq_map)));

    // 2. 合并处理
    std::vector<PairBytes> merges;
    const auto total_vocab_size = vocab_.size()+vocab_size_;
    merges.reserve(total_vocab_size);
    vocab_.reserve(total_vocab_size);
    while (vocab_.size() < vocab_size_) {
        // best pair
        const auto& best_pair = pairsfreq_queue_->pop();
        const auto vocabs_to_merge = pairs2vocab_map_[best_pair]; 
        pairs2vocab_map_.erase(best_pair);
        // 更新 pairs2vocab_map_, all_merging_list_和all_vocab_freq_map_
        for (const auto& voc_idx : vocabs_to_merge) {
            auto [new_pairs, new_voc, freq] = mergePair(voc_idx, best_pair);
            size_t new_voc_idx = all_vocab_list_.size();
            
            for (const auto& p : new_pairs) {
                pairs2vocab_map_[p].insert(new_voc_idx);
            }

            all_vocab_list_.push_back(new_voc);
            all_vocab_freq_map_.emplace(new_voc_idx, freq);
        }
        vocab_.emplace(vocab_.size(), pair_bytes(best_pair));
        merges.push_back(best_pair);
    }

    return std::make_tuple(vocab_, merges);
}

// pairsfreq_queue_ 和 pairs2vocab_map_
// voc_idx: 当前需要merging对所在的单词。
std::tuple<std::vector<PairBytes>, std::vector<Bytes>, int> 
BytePairEncoding::mergePair(size_t voc_idx, const PairBytes& best_pair) {
    
    const auto& voc  = all_vocab_list_[voc_idx];
    const auto freq = all_vocab_freq_map_[voc_idx];
    const auto best_bytes = pair_bytes(best_pair);
    const auto n = voc.size();

    std::vector<PairBytes> new_pairs;
    std::vector<Bytes> new_vocab;
    Bytes curr_elem, prev_elem;

    for(size_t i = 0; i < n; ) {
        PairBytes curr_pair;
        
        if (i+1 < n){
            curr_pair = std::make_pair(voc[i], voc[i+1]);
            pairs2vocab_map_[curr_pair].erase(voc_idx);
        }
        
        if (i+1 < n && curr_pair == best_pair) {
            if (i > 0) {
                pairsfreq_queue_->increase(std::make_pair(voc[i-1], voc[i]), -freq);
                pairsfreq_queue_->increase(std::make_pair(voc[i-1], best_bytes), freq);
            }

            if (i+2 < n) {
                pairsfreq_queue_->increase(std::make_pair(voc[i+1], voc[i+2]), -freq);
                pairsfreq_queue_->increase(std::make_pair(best_bytes, voc[i+2]), freq);
            }

            curr_elem = best_bytes;
            
            // 补删跳过的i
            if (i+2 < n){
                pairs2vocab_map_[std::make_pair(voc[i+1], voc[i+2])].erase(voc_idx);
            }
            i += 2;
        } else {
            curr_elem = voc[i];
            i += 1;
        }
        
        new_vocab.push_back(curr_elem);
        if (!prev_elem.empty())
            new_pairs.emplace_back(prev_elem, curr_elem);
        prev_elem = curr_elem;
    }

    return std::make_tuple(new_pairs, new_vocab, freq);
}

} //end namespace
