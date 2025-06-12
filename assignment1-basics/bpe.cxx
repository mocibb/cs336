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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bpe.hxx"

namespace py = pybind11;

struct Converter {
    
    inline static py::bytes toPython(const Bytes& b) {
        return py::bytes(reinterpret_cast<const char*>(b.data()), b.size());
    }

    inline static Bytes toCpp(py::bytes py_bytes) {
        char* buffer;
        ssize_t length;
    
        if (PyBytes_AsStringAndSize(py_bytes.ptr(), &buffer, &length)) {
            throw py::error_already_set();
        }
        
        return {buffer, buffer + length};
    }
};

PYBIND11_MODULE(bpe, m) {
    py::class_<BytePairEncoding>(m, "BytePairEncoding")
        .def(py::init<int, const std::vector<std::string>&>(),
             py::arg("vocab_size"),
             py::arg("special_tokens") = std::vector<std::string>{})
        
        .def("setVocabFreq", [](BytePairEncoding& self, const py::dict& py_vocab_freq) {
                std::unordered_map<Bytes, int> cpp_vocab_freq;
                cpp_vocab_freq.reserve(py_vocab_freq.size());
                
                for (auto item : py_vocab_freq) {
                    cpp_vocab_freq[Converter::toCpp(item.first.cast<py::bytes>())] = item.second.cast<int>();
                }

                self.setVocabFreq(std::move(cpp_vocab_freq));
            }, py::arg("vocab_freq"))

        .def("merge", [](BytePairEncoding& self) {            
            const auto& [cpp_vocab, cpp_merges] = self.merge();
            std::unordered_map<int, py::bytes> py_vocab;
            py_vocab.reserve(cpp_vocab.size());

            for (const auto& [k, v] : cpp_vocab) {
                py_vocab[static_cast<int>(k)] = Converter::toPython(v);
            }
            
            std::vector<std::tuple<py::bytes, py::bytes>> py_merges;
            py_merges.reserve(cpp_merges.size());
            for (const auto& pair : cpp_merges) {
                py_merges.emplace_back(
                    Converter::toPython(pair.first),
                    Converter::toPython(pair.second)
                );
            }
            
            return py::make_tuple(py_vocab, py_merges);
        });
}