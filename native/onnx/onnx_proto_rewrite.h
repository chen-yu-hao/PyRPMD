/* Minimal zero-dependency ONNX ModelProto wire scanner/rewriter.
 * SPDX-License-Identifier: MIT
 *
 * Rewrites standard ai.onnx float64 ScatterND nodes to a custom domain at
 * session load time when targeting CUDAExecutionProvider, leaving float32/int64
 * built-in ScatterND schemas and on-disk .onnx files untouched.
 */
#ifndef PYRPMD_ONNX_PROTO_REWRITE_H_
#define PYRPMD_ONNX_PROTO_REWRITE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pyrpmd_onnx {

struct ScatterNDRewriteResult {
    bool valid_wire = false;
    bool has_float64_scatter_nd = false;
    bool rewritten = false;
    std::vector<uint8_t> model_bytes;
};

namespace detail {

inline bool read_varint(const uint8_t*& p, const uint8_t* end, uint64_t& out) noexcept {
    out = 0;
    unsigned int shift = 0;
    while (p < end && shift < 64) {
        const uint8_t byte = *p++;
        out |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) return true;
        shift += 7;
    }
    return false;
}

inline bool read_length_delimited(const uint8_t*& p, const uint8_t* end,
                                  const uint8_t*& data, size_t& len) noexcept {
    uint64_t raw_len = 0;
    if (!read_varint(p, end, raw_len)) return false;
    if (raw_len > static_cast<uint64_t>(end - p)) return false;
    data = p;
    len = static_cast<size_t>(raw_len);
    p += len;
    return true;
}

inline bool skip_wire_value(const uint8_t*& p, const uint8_t* end, uint32_t wire_type) noexcept {
    switch (wire_type) {
        case 0: {
            uint64_t dummy = 0;
            return read_varint(p, end, dummy);
        }
        case 1: {
            if (end - p < 8) return false;
            p += 8;
            return true;
        }
        case 2: {
            const uint8_t* d = nullptr;
            size_t n = 0;
            return read_length_delimited(p, end, d, n);
        }
        case 5: {
            if (end - p < 4) return false;
            p += 4;
            return true;
        }
        default:
            return false;
    }
}

inline void append_varint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<uint8_t>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

inline void append_tag(std::vector<uint8_t>& out, uint32_t field_number, uint32_t wire_type) {
    append_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
}

inline void append_bytes_field(std::vector<uint8_t>& out, uint32_t field_number,
                               const uint8_t* data, size_t len) {
    append_tag(out, field_number, 2);
    append_varint(out, static_cast<uint64_t>(len));
    out.insert(out.end(), data, data + len);
}

inline void append_string_field(std::vector<uint8_t>& out, uint32_t field_number,
                                std::string_view text) {
    append_bytes_field(out, field_number,
                       reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

inline bool parse_tensor_proto(const uint8_t* p, const uint8_t* end,
                               std::string& name, int32_t& data_type) {
    name.clear();
    data_type = 0;
    while (p < end) {
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 2 && wt == 0) {
            uint64_t v = 0;
            if (!read_varint(p, end, v)) return false;
            data_type = static_cast<int32_t>(v);
        } else if (field == 8 && wt == 2) {
            const uint8_t* s = nullptr;
            size_t len = 0;
            if (!read_length_delimited(p, end, s, len)) return false;
            name.assign(reinterpret_cast<const char*>(s), len);
        } else {
            if (!skip_wire_value(p, end, wt)) return false;
        }
    }
    return true;
}

inline bool parse_type_proto_elem_type(const uint8_t* p, const uint8_t* end, int32_t& elem_type) {
    elem_type = 0;
    while (p < end) {
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 1 && wt == 2) {
            const uint8_t* tp = nullptr;
            size_t tlen = 0;
            if (!read_length_delimited(p, end, tp, tlen)) return false;
            const uint8_t* tend = tp + tlen;
            while (tp < tend) {
                uint64_t ttag = 0;
                if (!read_varint(tp, tend, ttag) || (ttag >> 3) == 0) return false;
                const uint32_t tf = static_cast<uint32_t>(ttag >> 3);
                const uint32_t tw = static_cast<uint32_t>(ttag & 0x7u);
                if (tf == 1 && tw == 0) {
                    uint64_t v = 0;
                    if (!read_varint(tp, tend, v)) return false;
                    elem_type = static_cast<int32_t>(v);
                } else {
                    if (!skip_wire_value(tp, tend, tw)) return false;
                }
            }
        } else {
            if (!skip_wire_value(p, end, wt)) return false;
        }
    }
    return true;
}

inline bool parse_value_info_proto(const uint8_t* p, const uint8_t* end,
                                   std::string& name, int32_t& elem_type,
                                   std::vector<uint8_t>& raw_type_proto) {
    name.clear();
    elem_type = 0;
    raw_type_proto.clear();
    while (p < end) {
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 1 && wt == 2) {
            const uint8_t* s = nullptr;
            size_t len = 0;
            if (!read_length_delimited(p, end, s, len)) return false;
            name.assign(reinterpret_cast<const char*>(s), len);
        } else if (field == 2 && wt == 2) {
            const uint8_t* tp = nullptr;
            size_t tlen = 0;
            if (!read_length_delimited(p, end, tp, tlen)) return false;
            raw_type_proto.assign(tp, tp + tlen);
            if (!parse_type_proto_elem_type(tp, tp + tlen, elem_type)) return false;
        } else {
            if (!skip_wire_value(p, end, wt)) return false;
        }
    }
    return true;
}

inline bool parse_attribute_proto(const uint8_t* p, const uint8_t* end,
                                  std::string& name, int64_t& int_val, bool& has_int,
                                  int32_t& tensor_dtype, bool& has_tensor) {
    name.clear();
    int_val = 0;
    has_int = false;
    tensor_dtype = 0;
    has_tensor = false;
    while (p < end) {
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 1 && wt == 2) {
            const uint8_t* s = nullptr;
            size_t len = 0;
            if (!read_length_delimited(p, end, s, len)) return false;
            name.assign(reinterpret_cast<const char*>(s), len);
        } else if (field == 3 && wt == 0) {
            uint64_t v = 0;
            if (!read_varint(p, end, v)) return false;
            int_val = static_cast<int64_t>(v);
            has_int = true;
        } else if (field == 5 && wt == 2) {
            const uint8_t* t = nullptr;
            size_t tlen = 0;
            if (!read_length_delimited(p, end, t, tlen)) return false;
            std::string dummy_name;
            if (!parse_tensor_proto(t, t + tlen, dummy_name, tensor_dtype)) return false;
            has_tensor = true;
        } else {
            if (!skip_wire_value(p, end, wt)) return false;
        }
    }
    return true;
}

inline bool is_first_input_dtype_preserving(std::string_view op) noexcept {
    static constexpr std::string_view kOps[] = {
        "Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten",
        "Expand", "Slice", "Gather", "GatherND", "GatherElements", "Pad", "Split",
        "Concat", "Tile", "ScatterND", "ScatterElements", "Add", "Sub", "Mul",
        "Div", "Pow", "Sum", "Min", "Max", "Mean", "Neg", "Abs", "Exp", "Log",
        "Sqrt", "Sin", "Cos", "Tanh", "Relu", "Erf", "Sigmoid", "Softplus",
        "Reciprocal", "Floor", "Ceil", "Clip", "Gemm", "MatMul", "Einsum",
        "CumSum", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
        "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
        "ReduceSumSquare",
    };
    for (const auto candidate : kOps) {
        if (op == candidate) return true;
    }
    return false;
}

inline bool parse_opset_import(const uint8_t* p, const uint8_t* end, std::string& domain) {
    domain.clear();
    while (p < end) {
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 1 && wt == 2) {
            const uint8_t* s = nullptr;
            size_t len = 0;
            if (!read_length_delimited(p, end, s, len)) return false;
            domain.assign(reinterpret_cast<const char*>(s), len);
        } else {
            if (!skip_wire_value(p, end, wt)) return false;
        }
    }
    return true;
}

inline bool rewrite_node_domain(const uint8_t* p, const uint8_t* end,
                                std::string_view new_domain,
                                std::vector<uint8_t>& out_node) {
    out_node.clear();
    while (p < end) {
        const uint8_t* field_start = p;
        uint64_t tag = 0;
        if (!read_varint(p, end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (!skip_wire_value(p, end, wt)) return false;
        if (field != 7) {
            out_node.insert(out_node.end(), field_start, p);
        }
    }
    append_string_field(out_node, 7, new_domain);
    return true;
}

inline bool process_graph_proto(const uint8_t* graph_begin, const uint8_t* graph_end,
                                bool enable_rewrite, std::string_view target_domain,
                                bool& found_f64_scatter, std::vector<uint8_t>& new_graph) {
    constexpr int32_t kOnnxDouble = 11;
    std::unordered_map<std::string, int32_t> dtypes;
    std::unordered_map<std::string, std::vector<uint8_t>> type_protos;
    std::unordered_set<std::string> declared_value_infos;

    // Pass 1: collect initializers, inputs, outputs, and value_infos.
    {
        const uint8_t* p = graph_begin;
        while (p < graph_end) {
            uint64_t tag = 0;
            if (!read_varint(p, graph_end, tag) || (tag >> 3) == 0) return false;
            const uint32_t field = static_cast<uint32_t>(tag >> 3);
            const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
            if ((field == 5 || field == 11 || field == 12 || field == 13) && wt == 2) {
                const uint8_t* sub = nullptr;
                size_t len = 0;
                if (!read_length_delimited(p, graph_end, sub, len)) return false;
                if (field == 5) {
                    std::string tname;
                    int32_t dt = 0;
                    if (!parse_tensor_proto(sub, sub + len, tname, dt)) return false;
                    if (!tname.empty() && dt != 0) dtypes[tname] = dt;
                } else {
                    std::string vname;
                    int32_t dt = 0;
                    std::vector<uint8_t> raw_tp;
                    if (!parse_value_info_proto(sub, sub + len, vname, dt, raw_tp)) return false;
                    if (!vname.empty()) {
                        declared_value_infos.insert(vname);
                        if (dt != 0) dtypes[vname] = dt;
                        if (!raw_tp.empty()) type_protos[vname] = std::move(raw_tp);
                    }
                }
            } else {
                if (!skip_wire_value(p, graph_end, wt)) return false;
            }
        }
    }

    // Pass 2: walk nodes in topological order, propagate types, and rewrite float64 ScatterND.
    found_f64_scatter = false;
    new_graph.clear();
    if (enable_rewrite) {
        new_graph.reserve(static_cast<size_t>(graph_end - graph_begin) + 256);
    }
    std::vector<std::vector<uint8_t>> extra_value_infos;

    const uint8_t* p = graph_begin;
    while (p < graph_end) {
        const uint8_t* field_start = p;
        uint64_t tag = 0;
        if (!read_varint(p, graph_end, tag) || (tag >> 3) == 0) return false;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);

        if (field == 1 && wt == 2) {
            const uint8_t* node_ptr = nullptr;
            size_t node_len = 0;
            if (!read_length_delimited(p, graph_end, node_ptr, node_len)) return false;

            std::vector<std::string> inputs, outputs;
            std::string op_type, domain;
            int32_t cast_to = 0;
            int32_t attr_value_dtype = 0;
            bool has_attr_value_dtype = false;

            const uint8_t* np = node_ptr;
            const uint8_t* nend = node_ptr + node_len;
            while (np < nend) {
                uint64_t ntag = 0;
                if (!read_varint(np, nend, ntag) || (ntag >> 3) == 0) return false;
                const uint32_t nf = static_cast<uint32_t>(ntag >> 3);
                const uint32_t nw = static_cast<uint32_t>(ntag & 0x7u);
                if ((nf == 1 || nf == 2 || nf == 4 || nf == 5 || nf == 7) && nw == 2) {
                    const uint8_t* s = nullptr;
                    size_t slen = 0;
                    if (!read_length_delimited(np, nend, s, slen)) return false;
                    if (nf == 1) {
                        inputs.emplace_back(reinterpret_cast<const char*>(s), slen);
                    } else if (nf == 2) {
                        outputs.emplace_back(reinterpret_cast<const char*>(s), slen);
                    } else if (nf == 4) {
                        op_type.assign(reinterpret_cast<const char*>(s), slen);
                    } else if (nf == 7) {
                        domain.assign(reinterpret_cast<const char*>(s), slen);
                    } else if (nf == 5) {
                        std::string aname;
                        int64_t ival = 0;
                        bool has_i = false, has_t = false;
                        int32_t tdt = 0;
                        if (!parse_attribute_proto(s, s + slen, aname, ival, has_i, tdt, has_t)) {
                            return false;
                        }
                        if (aname == "to" && has_i) cast_to = static_cast<int32_t>(ival);
                        if (aname == "value" && has_t) {
                            attr_value_dtype = tdt;
                            has_attr_value_dtype = true;
                        }
                    }
                } else {
                    if (!skip_wire_value(np, nend, nw)) return false;
                }
            }

            if ((domain.empty() || domain == "ai.onnx") && !outputs.empty()) {
                int32_t inferred = 0;
                if (op_type == "Cast") {
                    inferred = cast_to;
                } else if (op_type == "Constant") {
                    inferred = has_attr_value_dtype ? attr_value_dtype : 0;
                } else if (op_type == "ConstantOfShape") {
                    inferred = has_attr_value_dtype ? attr_value_dtype : 1;
                } else if (op_type == "Where" && inputs.size() >= 2) {
                    auto it = dtypes.find(inputs[1]);
                    if (it != dtypes.end()) inferred = it->second;
                } else if (is_first_input_dtype_preserving(op_type) && !inputs.empty()) {
                    auto it = dtypes.find(inputs[0]);
                    if (it != dtypes.end()) inferred = it->second;
                }
                if (inferred != 0) {
                    for (const auto& out_name : outputs) {
                        if (!out_name.empty() && dtypes.find(out_name) == dtypes.end()) {
                            dtypes[out_name] = inferred;
                        }
                    }
                }
                if ((op_type == "ScatterND" || op_type == "Identity") && !inputs.empty()) {
                    auto tp_it = type_protos.find(inputs[0]);
                    if (tp_it != type_protos.end() && !outputs[0].empty()
                        && type_protos.find(outputs[0]) == type_protos.end()) {
                        type_protos[outputs[0]] = tp_it->second;
                    }
                }
            }

            const auto in0_dt = inputs.empty() ? dtypes.end() : dtypes.find(inputs[0]);
            const bool is_f64_scatter =
                (domain.empty() || domain == "ai.onnx") && op_type == "ScatterND"
                && in0_dt != dtypes.end() && in0_dt->second == kOnnxDouble;

            if (is_f64_scatter) {
                found_f64_scatter = true;
                if (enable_rewrite) {
                    std::vector<uint8_t> rewritten_node;
                    if (!rewrite_node_domain(node_ptr, node_ptr + node_len, target_domain, rewritten_node)) {
                        return false;
                    }
                    append_bytes_field(new_graph, 1, rewritten_node.data(), rewritten_node.size());
                    if (!outputs.empty() && !outputs[0].empty()
                        && declared_value_infos.find(outputs[0]) == declared_value_infos.end()) {
                        auto tp_it = type_protos.find(inputs[0]);
                        if (tp_it != type_protos.end()) {
                            std::vector<uint8_t> vi;
                            append_string_field(vi, 1, outputs[0]);
                            append_bytes_field(vi, 2, tp_it->second.data(), tp_it->second.size());
                            extra_value_infos.push_back(std::move(vi));
                            declared_value_infos.insert(outputs[0]);
                        }
                    }
                    continue;
                }
            }
            if (enable_rewrite) {
                new_graph.insert(new_graph.end(), field_start, p);
            }
        } else {
            if (!skip_wire_value(p, graph_end, wt)) return false;
            if (enable_rewrite) {
                new_graph.insert(new_graph.end(), field_start, p);
            }
        }
    }

    if (enable_rewrite && found_f64_scatter) {
        for (const auto& vi : extra_value_infos) {
            append_bytes_field(new_graph, 13, vi.data(), vi.size());
        }
    }
    return true;
}

}  // namespace detail

inline ScatterNDRewriteResult rewrite_float64_scatter_nd_for_cuda(
    const uint8_t* model_data, size_t model_size,
    bool enable_rewrite, std::string_view target_domain) {
    ScatterNDRewriteResult res;
    const uint8_t* const end = model_data + model_size;
    const uint8_t* p = model_data;

    bool has_target_opset = false;
    bool seen_graph = false;
    std::vector<uint8_t> new_graph;

    while (p < end) {
        uint64_t tag = 0;
        if (!detail::read_varint(p, end, tag) || (tag >> 3) == 0) return res;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (field == 7 && wt == 2) {
            const uint8_t* g = nullptr;
            size_t glen = 0;
            if (!detail::read_length_delimited(p, end, g, glen)) return res;
            seen_graph = true;
            if (!detail::process_graph_proto(g, g + glen, enable_rewrite, target_domain,
                                             res.has_float64_scatter_nd, new_graph)) {
                return res;
            }
        } else if (field == 8 && wt == 2) {
            const uint8_t* op = nullptr;
            size_t oplen = 0;
            if (!detail::read_length_delimited(p, end, op, oplen)) return res;
            std::string dom;
            if (!detail::parse_opset_import(op, op + oplen, dom)) return res;
            if (dom == target_domain) has_target_opset = true;
        } else {
            if (!detail::skip_wire_value(p, end, wt)) return res;
        }
    }

    if (!seen_graph) return res;
    res.valid_wire = true;
    if (!enable_rewrite || !res.has_float64_scatter_nd) {
        return res;
    }

    res.model_bytes.reserve(model_size + 128);
    p = model_data;
    while (p < end) {
        const uint8_t* field_start = p;
        uint64_t tag = 0;
        if (!detail::read_varint(p, end, tag)) return ScatterNDRewriteResult{};
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 0x7u);
        if (!detail::skip_wire_value(p, end, wt)) return ScatterNDRewriteResult{};
        if (field == 7 && wt == 2) {
            detail::append_bytes_field(res.model_bytes, 7, new_graph.data(), new_graph.size());
        } else {
            res.model_bytes.insert(res.model_bytes.end(), field_start, p);
        }
    }
    if (!has_target_opset) {
        std::vector<uint8_t> opset;
        detail::append_string_field(opset, 1, target_domain);
        detail::append_tag(opset, 2, 0);
        detail::append_varint(opset, 1);
        detail::append_bytes_field(res.model_bytes, 8, opset.data(), opset.size());
    }
    res.rewritten = true;
    return res;
}

}  // namespace pyrpmd_onnx

#endif  // PYRPMD_ONNX_PROTO_REWRITE_H_
