#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include "ksp_qp_types.hpp"

template <typename T>
class MpsFormatParser {
public:
    using SpMat = Eigen::SparseMatrix<T>;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    ParsedModel<T> parse(const std::string& filename);
    KSPQPdata<T> to_kspqp(const ParsedModel<T>& model, T eq_tol = 1e-12, T inf_cap = std::numeric_limits<T>::infinity());

private:
    enum class Section {
        NONE, NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ, ENDATA
    };

    struct RowInfo {
        // For ROWS section
        char type = 'N'; // 'N', 'E', 'L', 'G'
        int idx   = -1;  // index of constraint rows (objective row has idx = -1)
    };

    // Fixed-vs-free-format: decided once from the first value-bearing line.
    enum class Format { UNKNOWN, FIXED, FREE };

    Section section_ = Section::NONE;
    Format format_ = Format::UNKNOWN;
    std::string sense_ = "MIN";

    std::string obj_name_;
    std::unordered_map<std::string, RowInfo> row_map_;
    std::unordered_map<std::string, int> col_map_;

    std::vector<char> col_lower_explicit_;

    std::vector<T> rhs_values_;
    std::vector<T> range_values_;

    std::vector<Triplet> A_triplets_;
    std::vector<Triplet> Q_triplets_;

    ParsedModel<T> model_;

    // ws_tokens_ holds the current line's raw whitespace split,
    // tokens_ holds the final section-specific tokens handed to parse_*().
    // Both are reused for each line, so they are only valid until the next getline() call.
    std::vector<std::string_view> ws_tokens_;
    std::vector<std::string_view> tokens_;

    static bool is_comment_or_blank(const std::string& line);
    static void split_ws(std::string_view line, std::vector<std::string_view>& out);
    static void split_free_by_section(const std::vector<std::string_view>& toks, Section sec,
                                       std::vector<std::string_view>& out);
    static void split_fixed_by_section(std::string_view line, Section sec,
                                        std::vector<std::string_view>& out);
    void tokenize_line(const std::string& line, Section sec);
    static bool fixed_fields_would_truncate(std::string_view line,
                                             const std::vector<std::string_view>& toks);
    static std::string_view trim(std::string_view s);

    bool set_section(const std::vector<std::string_view>& tokens);
    void decide_format_from(const std::string& line, Section sec);
    void parse_objsense(const std::vector<std::string_view>& tokens);
    void parse_rows(const std::vector<std::string_view>& tokens);
    void parse_columns(const std::vector<std::string_view>& tokens);
    void parse_rhs(const std::vector<std::string_view>& tokens);
    void parse_ranges(const std::vector<std::string_view>& tokens);
    void parse_bounds(const std::vector<std::string_view>& tokens);
    void parse_quadobj(const std::vector<std::string_view>& tokens);

    void finalize_defaults();
    void finalize_row_bounds();
    void build_sparse_matrices();
};

#include "mps_format_parser.tpp"