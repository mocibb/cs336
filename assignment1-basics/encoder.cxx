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
#include <pybind11/stl_bind.h> 
#include <pybind11/stl.h>
#include "encoder.hxx"

namespace py = pybind11;

struct Converter {

    inline static Bytes toCpp(py::bytes py_bytes) {
        char* buffer;
        ssize_t length;
    
        if (PyBytes_AsStringAndSize(py_bytes.ptr(), &buffer, &length)) {
            throw py::error_already_set();
        }
        
        return {buffer, buffer + length};
    }
};

PYBIND11_MODULE(encoder, m) {
    // 绑定Bytes类型
    py::bind_vector<Bytes>(m, "Bytes");
    
    // 绑定unordered_map<Bytes, int>类型
    py::bind_map<std::unordered_map<Bytes, int>>(m, "VocabMap");

    py::class_<Encoder>(m, "Encoder")
        .def(py::init([](const py::dict& vocab_dict, int cache_size) {
            std::unordered_map<Bytes, int> vocab_map;
            for (auto item : vocab_dict) {
                try {
                    Bytes key = Converter::toCpp(item.first.cast<py::bytes>());
                    
                    // 转换值：Python int -> C++ int
                    int value = item.second.cast<int>();
                    
                    vocab_map.emplace(std::move(key), value);
                } catch (const py::cast_error& e) {
                    throw py::value_error("Invalid type in vocab dictionary. "
                                         "Keys must be bytes, values must be integers.");
                }
            }
            return new Encoder(std::move(vocab_map), cache_size);
        }), py::arg("vocab2ids"), py::arg("cache_size") = 10000)
        .def("_encode", &Encoder::_encode, py::arg("text"));
}