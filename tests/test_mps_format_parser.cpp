#include "mps_format_parser.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <utility>

namespace {

constexpr double kTight = 1e-12;

// Build one line of fixed-format MPS data (F1@2, F2@5, F3@15, F4@25, F5@40, F6@50).
std::string MpsLine(std::initializer_list<std::pair<int, std::string>> fields) {
  std::string line;
  for (const auto& [col1, text] : fields) {
    const int col0 = col1 - 1;
    if ((int)line.size() < col0) line.resize(col0, ' ');
    line += text;
  }
  return line;
}

// Build one line of free-format MPS data (whitespace-delimited, non-column-aligned).
std::string FreeLine(std::initializer_list<std::string> fields) {
  std::string line;
  bool first = true;
  for (const auto& f : fields) {
    if (!first) line += " ";
    line += f;
    first = false;
  }
  return line;
}

std::string WriteTempMps(const std::string& content) {
  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string name =
      std::string(test_info->test_suite_name()) + "_" + test_info->name() + ".mps";
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream f(path);
  f << content;
  return path.string();
}

}  // namespace

// ===================== parse(): basic LP structure =====================

TEST(MpsFormatParserParse, BasicLpRowsColumnsRhs) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "L"}, {5, "LIM1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "LIM1"}, {50, "2.0"}}) + "\n";
  content += MpsLine({{5, "X2"}, {15, "COST"}, {25, "2.0"}, {40, "LIM1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "LIM1"}, {25, "10.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));

  EXPECT_EQ(model.num_cols, 2);
  EXPECT_EQ(model.num_rows, 1);
  EXPECT_FALSE(model.is_qp);
  EXPECT_TRUE(model.is_min);  // no OBJSENSE section => default MIN

  EXPECT_NEAR(model.c(0), 1.0, kTight);
  EXPECT_NEAR(model.c(1), 2.0, kTight);
  EXPECT_NEAR(model.A.coeff(0, 0), 2.0, kTight);
  EXPECT_NEAR(model.A.coeff(0, 1), 1.0, kTight);

  // L row with no RANGES entry: (-inf, rhs].
  EXPECT_EQ(model.row_upper(0), 10.0);
  EXPECT_EQ(model.row_lower(0), -std::numeric_limits<double>::infinity());

  // No BOUNDS section => default [0, inf) on both variables.
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(model.col_lower(i), 0.0);
    EXPECT_EQ(model.col_upper(i), std::numeric_limits<double>::infinity());
  }
}

TEST(MpsFormatParserParse, DuplicateColumnsEntriesForSameRowAreSummed) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  // Two separate COLUMNS entries for the same (X1, C1) pair must accumulate.
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "3.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "C1"}, {25, "4.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "0.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_NEAR(model.A.coeff(0, 0), 7.0, kTight);  // 3.0 + 4.0
}

// Degenerate minimal model: no constraint rows and no variables at all
// (just the mandatory objective row).
TEST(MpsFormatParserParse, EmptyModelWithNoRowsOrColumnsParsesCleanly) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_EQ(model.num_rows, 0);
  EXPECT_EQ(model.num_cols, 0);
  EXPECT_EQ(model.A.rows(), 0);
  EXPECT_EQ(model.A.cols(), 0);
  EXPECT_EQ(model.c.size(), 0);

  auto pd = parser.to_kspqp(model);
  EXPECT_EQ(pd.n, 0);
  EXPECT_EQ(pd.m, 0);
  EXPECT_EQ(pd.l, 0);
}

// ===================== parse(): fixed vs. free format detection =====================
// Free-format ROWS line "N GOAL" must not be read as fixed-format, where it slices to
// F1="G", F2="AL" (2 tokens, which passes the ROWS arity check).

TEST(MpsFormatParserParse, FreeFormatRowNamedGoalIsNotMisreadAsGTypeRowNamedAl) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += FreeLine({"N", "GOAL"}) + "\n";
  content += FreeLine({"L", "LIM1"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0", "LIM1", "1.0"}) + "\n";
  content += "RHS\n";
  content += FreeLine({"RHS", "LIM1", "5.0"}) + "\n";
  content += FreeLine({"RHS", "GOAL", "0.0"}) + "\n"; // Only resolves if ROWS correctly registered "GOAL".
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  ParsedModel<double> model;
  EXPECT_NO_THROW(model = parser.parse(WriteTempMps(content)));

  ASSERT_EQ(model.num_rows, 2);  // LIM1 + the free "GOAL" row
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(model.row_lower(0), -inf);  // GOAL: free row
  EXPECT_EQ(model.row_upper(0), inf);
  EXPECT_NEAR(model.row_upper(1), 5.0, kTight);  // LIM1
}

// ===================== parse(): free-format coverage =====================
// Free-format twins of the fixed-format tests above.

TEST(MpsFormatParserParse, FreeFormatRowsColumnsRhsMatchFixedFormatResult) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += FreeLine({"L", "LIM1"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0", "LIM1", "2.0"}) + "\n";
  content += FreeLine({"X2", "COST", "2.0", "LIM1", "1.0"}) + "\n";
  content += "RHS\n";
  content += FreeLine({"RHS", "LIM1", "10.0"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));

  EXPECT_EQ(model.num_cols, 2);
  EXPECT_EQ(model.num_rows, 1);
  EXPECT_FALSE(model.is_qp);
  EXPECT_TRUE(model.is_min);

  EXPECT_NEAR(model.c(0), 1.0, kTight);
  EXPECT_NEAR(model.c(1), 2.0, kTight);
  EXPECT_NEAR(model.A.coeff(0, 0), 2.0, kTight);
  EXPECT_NEAR(model.A.coeff(0, 1), 1.0, kTight);

  EXPECT_EQ(model.row_upper(0), 10.0);
  EXPECT_EQ(model.row_lower(0), -std::numeric_limits<double>::infinity());

  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(model.col_lower(i), 0.0);
    EXPECT_EQ(model.col_upper(i), std::numeric_limits<double>::infinity());
  }
}

TEST(MpsFormatParserParse, FreeFormatRangesAppliesToAllRowTypes) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += FreeLine({"E", "R1"}) + "\n";
  content += FreeLine({"L", "R2"}) + "\n";
  content += FreeLine({"G", "R3"}) + "\n";
  content += FreeLine({"N", "R4"}) + "\n";  // second N row: free constraint
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0", "R1", "1.0"}) + "\n";
  content += FreeLine({"X1", "R2", "1.0", "R3", "1.0"}) + "\n";
  content += FreeLine({"X1", "R4", "1.0"}) + "\n";
  content += "RHS\n";
  content += FreeLine({"RHS", "R1", "5.0", "R2", "8.0"}) + "\n";
  content += FreeLine({"RHS", "R3", "2.0"}) + "\n";
  content += "RANGES\n";
  content += FreeLine({"RNG", "R1", "3.0", "R2", "2.0"}) + "\n";
  content += FreeLine({"RNG", "R3", "4.0"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_rows, 4);

  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_NEAR(model.row_lower(0), 2.0, kTight);  // R1 (E, rhs=5, range=3): [2,8]
  EXPECT_NEAR(model.row_upper(0), 8.0, kTight);
  EXPECT_NEAR(model.row_lower(1), 6.0, kTight);  // R2 (L, rhs=8, range=2): [6,8]
  EXPECT_NEAR(model.row_upper(1), 8.0, kTight);
  EXPECT_NEAR(model.row_lower(2), 2.0, kTight);  // R3 (G, rhs=2, range=4): [2,6]
  EXPECT_NEAR(model.row_upper(2), 6.0, kTight);
  EXPECT_EQ(model.row_lower(3), -inf);  // R4: free
  EXPECT_EQ(model.row_upper(3), inf);
}

TEST(MpsFormatParserParse, FreeFormatBoundsAllTypesAndDefaultBoundName) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += FreeLine({"G", "C1"}) + "\n";
  content += "COLUMNS\n";
  for (const std::string& v : {"X1", "X2", "X3", "X4", "X5", "X6"}) {
    content += FreeLine({v, "COST", "1.0", "C1", "1.0"}) + "\n";
  }
  content += "RHS\n";
  content += FreeLine({"RHS", "C1", "0.0"}) + "\n";
  content += "BOUNDS\n";
  content += FreeLine({"LO", "BND", "X1", "-5.0"}) + "\n";
  content += FreeLine({"UP", "BND", "X1", "5.0"}) + "\n";
  content += FreeLine({"FX", "BND", "X2", "3.0"}) + "\n";
  content += FreeLine({"FR", "BND", "X3"}) + "\n";
  content += FreeLine({"MI", "BND", "X4"}) + "\n";
  content += FreeLine({"PL", "BND", "X4"}) + "\n";
  content += FreeLine({"BV", "BND", "X5"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_cols, 6);
  const double inf = std::numeric_limits<double>::infinity();

  EXPECT_NEAR(model.col_lower(0), -5.0, kTight);  // X1: LO then UP
  EXPECT_NEAR(model.col_upper(0), 5.0, kTight);
  EXPECT_NEAR(model.col_lower(1), 3.0, kTight);  // X2: FX
  EXPECT_NEAR(model.col_upper(1), 3.0, kTight);
  EXPECT_EQ(model.col_lower(2), -inf);  // X3: FR
  EXPECT_EQ(model.col_upper(2), inf);
  EXPECT_EQ(model.col_lower(3), -inf);  // X4: MI then PL
  EXPECT_EQ(model.col_upper(3), inf);
  EXPECT_NEAR(model.col_lower(4), 0.0, kTight);  // X5: BV
  EXPECT_NEAR(model.col_upper(4), 1.0, kTight);
  EXPECT_NEAR(model.col_lower(5), 0.0, kTight);  // X6: untouched, default bound
  EXPECT_EQ(model.col_upper(5), inf);
}

TEST(MpsFormatParserParse, FreeFormatQuadobjThreeTokenFormStoresLowerTriangleOnly) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += FreeLine({"G", "C1"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0", "C1", "1.0"}) + "\n";
  content += FreeLine({"X2", "COST", "1.0", "C1", "1.0"}) + "\n";
  content += "RHS\n";
  content += FreeLine({"RHS", "C1", "0.0"}) + "\n";
  content += "QUADOBJ\n";
  content += FreeLine({"X1", "X1", "4.0"}) + "\n";
  content += FreeLine({"X1", "X2", "1.0"}) + "\n";  // off-diagonal
  content += FreeLine({"X2", "X2", "2.0"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_TRUE(model.is_qp);

  EXPECT_NEAR(model.Q.coeff(0, 0), 4.0, kTight);
  EXPECT_NEAR(model.Q.coeff(1, 0), 1.0, kTight);
  EXPECT_NEAR(model.Q.coeff(1, 1), 2.0, kTight);
  EXPECT_NEAR(model.Q.coeff(0, 1), 0.0, kTight);  // upper triangle NOT populated
}

// No fixed-format analog: exercises split_free_by_section()'s QUADOBJ
// 4-token ("name col1 col2 val") branch, which drops the leading name.
TEST(MpsFormatParserParse, FreeFormatQuadobjFourTokenNamePrefixedFormDropsLeadingName) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0"}) + "\n";
  content += "QUADOBJ\n";
  content += FreeLine({"QNAME", "X1", "X1", "4.0"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_TRUE(model.is_qp);
  EXPECT_NEAR(model.Q.coeff(0, 0), 4.0, kTight);
}

// Row/column names longer than the 8-char fixed-field width:
// only representable in free format (fixed would truncate them).
TEST(MpsFormatParserParse, FreeFormatLongRowAndColumnNamesBeyondEightCharsAreSupported) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "OBJECTIVE_ROW"}) + "\n";
  content += FreeLine({"L", "CONSTRAINT_ONE"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"VARIABLE_ONE", "OBJECTIVE_ROW", "1.0", "CONSTRAINT_ONE", "2.0"}) + "\n";
  content += "RHS\n";
  content += FreeLine({"RHS", "CONSTRAINT_ONE", "10.0"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_cols, 1);
  ASSERT_EQ(model.num_rows, 1);
  EXPECT_NEAR(model.c(0), 1.0, kTight);
  EXPECT_NEAR(model.A.coeff(0, 0), 2.0, kTight);
  EXPECT_NEAR(model.row_upper(0), 10.0, kTight);
}

// ===================== parse(): OBJSENSE =====================

TEST(MpsFormatParserParse, ObjsenseMaxNegatesCorrectly) {
  std::string content;
  content += "OBJSENSE\n";
  content += "    MAX\n";
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "0.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_FALSE(model.is_min);

  auto pd = parser.to_kspqp(model);
  EXPECT_NEAR(pd.c(0), -1.0, kTight);  // negated because MAX -> MIN conversion
}

TEST(MpsFormatParserParse, ObjsenseUnknownSenseThrows) {
  std::string content;
  content += "OBJSENSE\n";
  content += "    BOGUS\n";
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

// ===================== parse(): row types E/L/G/N + RANGES =====================

TEST(MpsFormatParserParse, RowBoundsForEachTypeWithRanges) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "E"}, {5, "R1"}}) + "\n";
  content += MpsLine({{2, "L"}, {5, "R2"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "R3"}}) + "\n";
  content += MpsLine({{2, "N"}, {5, "R4"}}) + "\n";  // second N row: free constraint
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "R1"}, {50, "1.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "R2"}, {25, "1.0"}, {40, "R3"}, {50, "1.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "R4"}, {25, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "R1"}, {25, "5.0"}, {40, "R2"}, {50, "8.0"}}) + "\n";
  content += MpsLine({{5, "RHS"}, {15, "R3"}, {25, "2.0"}}) + "\n";
  content += "RANGES\n";
  content += MpsLine({{5, "RNG"}, {15, "R1"}, {25, "3.0"}, {40, "R2"}, {50, "2.0"}}) + "\n";
  content += MpsLine({{5, "RNG"}, {15, "R3"}, {25, "4.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_rows, 4);

  const double inf = std::numeric_limits<double>::infinity();
  // R1 (E, rhs=5, range=3): [5-3, 5+3]
  EXPECT_NEAR(model.row_lower(0), 2.0, kTight);
  EXPECT_NEAR(model.row_upper(0), 8.0, kTight);
  // R2 (L, rhs=8, range=2): [8-2, 8]
  EXPECT_NEAR(model.row_lower(1), 6.0, kTight);
  EXPECT_NEAR(model.row_upper(1), 8.0, kTight);
  // R3 (G, rhs=2, range=4): [2, 2+4]
  EXPECT_NEAR(model.row_lower(2), 2.0, kTight);
  EXPECT_NEAR(model.row_upper(2), 6.0, kTight);
  // R4 (second N row): free constraint, ignores RHS entirely.
  EXPECT_EQ(model.row_lower(3), -inf);
  EXPECT_EQ(model.row_upper(3), inf);
}

// finalize_row_bounds() takes std::abs(range_values_[i]), so the sign of a
// RANGES entry must not affect the resulting bounds.
TEST(MpsFormatParserParse, NegativeRangesMagnitudeIsAbsolutedForBoundsComputation) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "E"}, {5, "R1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "R1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "R1"}, {25, "5.0"}}) + "\n";
  content += "RANGES\n";
  content += MpsLine({{5, "RNG"}, {15, "R1"}, {25, "-3.0"}}) + "\n";  // negative range
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  // Same bounds as a positive range of the same magnitude: [5-3, 5+3].
  EXPECT_NEAR(model.row_lower(0), 2.0, kTight);
  EXPECT_NEAR(model.row_upper(0), 8.0, kTight);
}

// ===================== parse(): BOUNDS types =====================

TEST(MpsFormatParserParse, AllBoundTypesAndDefaultBound) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  for (const std::string& v : {"X1", "X2", "X3", "X4", "X5", "X6"}) {
    content += MpsLine({{5, v}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  }
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "0.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "-5.0"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "5.0"}}) + "\n";
  content += MpsLine({{2, "FX"}, {5, "BND"}, {15, "X2"}, {25, "3.0"}}) + "\n";
  content += MpsLine({{2, "FR"}, {5, "BND"}, {15, "X3"}}) + "\n";
  content += MpsLine({{2, "MI"}, {5, "BND"}, {15, "X4"}}) + "\n";
  content += MpsLine({{2, "PL"}, {5, "BND"}, {15, "X4"}}) + "\n";
  content += MpsLine({{2, "BV"}, {5, "BND"}, {15, "X5"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_cols, 6);
  const double inf = std::numeric_limits<double>::infinity();

  EXPECT_NEAR(model.col_lower(0), -5.0, kTight);  // X1: LO then UP
  EXPECT_NEAR(model.col_upper(0), 5.0, kTight);
  EXPECT_NEAR(model.col_lower(1), 3.0, kTight);  // X2: FX
  EXPECT_NEAR(model.col_upper(1), 3.0, kTight);
  EXPECT_EQ(model.col_lower(2), -inf);  // X3: FR
  EXPECT_EQ(model.col_upper(2), inf);
  EXPECT_EQ(model.col_lower(3), -inf);  // X4: MI then PL
  EXPECT_EQ(model.col_upper(3), inf);
  EXPECT_NEAR(model.col_lower(4), 0.0, kTight);  // X5: BV
  EXPECT_NEAR(model.col_upper(4), 1.0, kTight);
  EXPECT_NEAR(model.col_lower(5), 0.0, kTight);  // X6: untouched, default bound
  EXPECT_EQ(model.col_upper(5), inf);
}

TEST(MpsFormatParserParse, SecondBoundsSetWithDifferentNameStillApplies) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND1"}, {15, "X1"}, {25, "-5.0"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND2"}, {15, "X1"}, {25, "9.0"}}) + "\n";  // different name
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_NEAR(model.col_lower(0), -5.0, kTight);  // from BND1
  EXPECT_NEAR(model.col_upper(0), 9.0, kTight);    // from BND2, still applied
}

// A value-less bound (FR/MI/PL/BV) on numerically named columns, as in DPKLO1.SIF (columns
// "1".."133", bound set "0"). "FR 0 1" must read as <type> <bound_name> <col>; reading "1" as a
// value would leave every real column at [0, inf) and add a spurious free column.
TEST(MpsFormatParserParse, ValuelessBoundOnNumericallyNamedColumnIsNotMisreadAsAValue) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "78"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "1"}, {15, "78"}, {25, "1.0"}}) + "\n";
  content += MpsLine({{5, "2"}, {15, "78"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "FR"}, {5, "0"}, {15, "1"}}) + "\n";
  content += MpsLine({{2, "MI"}, {5, "0"}, {15, "2"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  const double inf = std::numeric_limits<double>::infinity();

  ASSERT_EQ(model.num_cols, 2);  // no spurious column named "0"
  EXPECT_EQ(model.col_lower(0), -inf);  // column "1": FR
  EXPECT_EQ(model.col_upper(0), inf);
  EXPECT_EQ(model.col_lower(1), -inf);  // column "2": MI
  EXPECT_EQ(model.col_upper(1), inf);
}

// The bound set name may be omitted, leaving a value-less type with a trailing
// (ignored) value.  Only then may the middle token be the column name.
TEST(MpsFormatParserParse, ValuelessBoundWithOmittedBoundNameAndTrailingValueUsesMiddleToken) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0"}) + "\n";
  content += "BOUNDS\n";
  content += FreeLine({"MI", "X1", "-1.0e30"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  const double inf = std::numeric_limits<double>::infinity();

  ASSERT_EQ(model.num_cols, 1);  // no spurious column named "-1.0e30"
  EXPECT_EQ(model.col_lower(0), -inf);
  EXPECT_EQ(model.col_upper(0), inf);
}

// A column that appears only in BOUNDS (an all-zero column) is still created
// from the third token rather than being confused with the bound set name.
TEST(MpsFormatParserParse, ValuelessBoundCreatesColumnSeenOnlyInBoundsSection) {
  std::string content;
  content += "ROWS\n";
  content += FreeLine({"N", "COST"}) + "\n";
  content += "COLUMNS\n";
  content += FreeLine({"X1", "COST", "1.0"}) + "\n";
  content += "BOUNDS\n";
  content += FreeLine({"FR", "BND", "X2"}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  const double inf = std::numeric_limits<double>::infinity();

  ASSERT_EQ(model.num_cols, 2);  // X1 and X2, not a column named "BND"
  EXPECT_EQ(model.col_lower(1), -inf);
  EXPECT_EQ(model.col_upper(1), inf);
}

TEST(MpsFormatParserParse, UnknownBoundTypeThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "ZZ"}, {5, "BND"}, {15, "X1"}, {25, "1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

// ===================== parse(): fixed-field overflow =====================

// Fixed-format by its first line (" N  OBJ" reads the same both ways), but with values at full
// double precision: "1.1305249478260869e+01" overruns the 12-column value field and would be
// read as "1.1305249478". Every value must survive intact.
TEST(MpsFormatParserParse, FullPrecisionValuesOverflowingTheFixedValueFieldAreNotTruncated) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "OBJ"}}) + "\n";
  content += MpsLine({{2, "L"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "OBJ"}, {25, "7.6099999999999994e+00"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "C1"}, {25, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "3.5519999999999996e-02"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "5.4412913913043628e+00"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "1.1305249478260869e+01"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));

  EXPECT_NEAR(model.c(0), 7.6099999999999994e+00, kTight);
  EXPECT_NEAR(model.row_upper(0), 3.5519999999999996e-02, kTight);
  EXPECT_NEAR(model.col_lower(0), 5.4412913913043628e+00, kTight);
  EXPECT_NEAR(model.col_upper(0), 1.1305249478260869e+01, kTight);  // not 1.1305249478
}

// Truncated, the upper bound (1.13) would fall below the lower bound (5.44) and throw an
// inconsistent-bounds error; it must not.
TEST(MpsFormatParserParse, OverflowingBoundPairNoLongerTriggersInconsistentBoundsError) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "OBJ"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "OBJ"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "5.4412913913043628e+00"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "1.1305249478260869e+01"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_NO_THROW(parser.parse(WriteTempMps(content)));
}

// The overflow override is per-line: short values on the same file still take the
// fixed reading, so a file mixing both widths parses correctly throughout.
TEST(MpsFormatParserParse, ShortAndOverflowingValuesCoexistInOneFixedFormatFile) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "OBJ"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "OBJ"}, {25, "40"}}) + "\n";
  content += MpsLine({{5, "X2"}, {15, "OBJ"}, {25, "5.2533480000000001e+01"}}) + "\n";
  content += MpsLine({{5, "X3"}, {15, "OBJ"}, {25, "2.6"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  ASSERT_EQ(model.num_cols, 3);
  EXPECT_NEAR(model.c(0), 40.0, kTight);
  EXPECT_NEAR(model.c(1), 5.2533480000000001e+01, kTight);
  EXPECT_NEAR(model.c(2), 2.6, kTight);
}

// ===================== parse(): QUADOBJ / lower-triangular storage =====================

TEST(MpsFormatParserParse, QuadobjStoresLowerTriangleOnly) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  content += MpsLine({{5, "X2"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "0.0"}}) + "\n";
  content += "QUADOBJ\n";
  content += MpsLine({{5, "X1"}, {15, "X1"}, {25, "4.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "X2"}, {25, "1.0"}}) + "\n";  // off-diagonal
  content += MpsLine({{5, "X2"}, {15, "X2"}, {25, "2.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_TRUE(model.is_qp);

  // X1=col0, X2=col1. Entry (X1,X2) has i=0 < j=1, so parse_quadobj swaps
  // to store it at (row=1, col=0) -- only the lower triangle is populated.
  EXPECT_NEAR(model.Q.coeff(0, 0), 4.0, kTight);
  EXPECT_NEAR(model.Q.coeff(1, 0), 1.0, kTight);
  EXPECT_NEAR(model.Q.coeff(1, 1), 2.0, kTight);
  EXPECT_NEAR(model.Q.coeff(0, 1), 0.0, kTight);  // upper triangle NOT populated
}

// Two separate QUADOBJ lines for the same (X1,X1) diagonal pair must accumulate.
TEST(MpsFormatParserParse, DuplicateQuadobjDiagonalEntriesAreSummed) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "QUADOBJ\n";
  content += MpsLine({{5, "X1"}, {15, "X1"}, {25, "4.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "X1"}, {25, "3.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_NEAR(model.Q.coeff(0, 0), 7.0, kTight);  // 4.0 + 3.0
}

// ===================== parse(): error paths =====================

TEST(MpsFormatParserParse, MissingFileThrows) {
  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse("/nonexistent/path/does_not_exist.mps"), std::runtime_error);
}

TEST(MpsFormatParserParse, DuplicateRowNameThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += MpsLine({{2, "L"}, {5, "C1"}}) + "\n";  // duplicate name
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, ColumnsReferencingUndefinedRowThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "NOPE"}, {25, "1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, UnknownRowTypeThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "X"}, {5, "R1"}}) + "\n";  // not E/L/G/N
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "R1"}, {50, "1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, MalformedRowsLineThrows) {
  std::string content;
  content += "ROWS\n";
  content += "N\n";  // missing row name: too few tokens
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, MalformedColumnsLineThrows) {
  std::string content;
  content += "ROWS\n";
  content += "N COST\n";
  content += "COLUMNS\n";
  content += "X1\n";  // column name with no (row, value) pair
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, MalformedRhsLineThrows) {
  std::string content;
  content += "ROWS\n";
  content += "N COST\n";
  content += "G C1\n";
  content += "COLUMNS\n";
  content += "X1 COST 1.0 C1 1.0\n";
  content += "RHS\n";
  content += "BADLINE\n";  // single token: too few for any RHS line shape
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, MalformedRangesLineThrows) {
  std::string content;
  content += "ROWS\n";
  content += "N COST\n";
  content += "E R1\n";
  content += "COLUMNS\n";
  content += "X1 COST 1.0 R1 1.0\n";
  content += "RHS\n";
  content += "RHS R1 5.0\n";
  content += "RANGES\n";
  content += "BAD LINE\n";  // 2 tokens: too few for RANGES (needs >= 3)
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, MalformedBoundsLineThrows) {
  std::string content;
  content += "ROWS\n";
  content += "N COST\n";
  content += "COLUMNS\n";
  content += "X1 COST 1.0\n";
  content += "BOUNDS\n";
  content += "UP\n";  // single token: missing column name entirely
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

TEST(MpsFormatParserParse, InconsistentColumnBoundsThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "5.0"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "1.0"}}) + "\n";  // < lower
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}


// A negative UP with no explicit LO relaxes the lower bound to -inf (the common MPS convention)
// instead of throwing on lower(0) > upper(<0).
TEST(MpsFormatParserParse, UpBoundNegativeWithoutLowerBoundRelaxesLowerBoundToMinusInf) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "-1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  ParsedModel<double> model;
  EXPECT_NO_THROW(model = parser.parse(WriteTempMps(content)));

  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(model.col_lower(0), -inf);
  EXPECT_NEAR(model.col_upper(0), -1.0, kTight);
}

// An explicit LO (even LO 0) disables the relaxation, so a following negative UP must throw.
TEST(MpsFormatParserParse, UpBoundNegativeWithExplicitZeroLowerBoundStillThrows) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "0.0"}}) + "\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "-1.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  EXPECT_THROW(parser.parse(WriteTempMps(content)), std::runtime_error);
}

// The relaxation checks for an explicit lower bound anywhere in the file, not only before the UP:
// a negative UP followed by a consistent LO keeps the LO value.
TEST(MpsFormatParserParse, UpBoundNegativeBeforeExplicitLowerBoundKeepsExplicitLowerBound) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}}) + "\n";
  content += "BOUNDS\n";
  content += MpsLine({{2, "UP"}, {5, "BND"}, {15, "X1"}, {25, "-1.0"}}) + "\n";
  content += MpsLine({{2, "LO"}, {5, "BND"}, {15, "X1"}, {25, "-5.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));

  EXPECT_NEAR(model.col_lower(0), -5.0, kTight);  // explicit LO, not the -inf relaxation
  EXPECT_NEAR(model.col_upper(0), -1.0, kTight);
}

TEST(MpsFormatParserParse, RhsSecondSetWithDifferentNameStillApplies) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C2"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "C2"}, {25, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS1"}, {15, "C1"}, {25, "5.0"}}) + "\n";
  content += MpsLine({{5, "RHS2"}, {15, "C2"}, {25, "3.0"}}) + "\n";  // different name
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_NEAR(model.row_lower(0), 5.0, kTight);  // C1, from RHS1
  EXPECT_NEAR(model.row_lower(1), 3.0, kTight);  // C2, from RHS2, still applied
}

// An RHS entry naming the objective (N) row adds to obj_const.
TEST(MpsFormatParserParse, RhsEntryTargetingObjectiveRowSetsObjConst) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "G"}, {5, "C1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "C1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "COST"}, {25, "7.0"}}) + "\n";
  content += MpsLine({{5, "RHS"}, {15, "C1"}, {25, "0.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  EXPECT_NEAR(model.obj_const, 7.0, kTight);

  auto pd = parser.to_kspqp(model);
  EXPECT_NEAR(pd.obj_const, -7.0, kTight);  // negated in to_kspqp(), read off the RHS
}

// A RANGES entry on a non-objective N row is accepted and ignored: finalize_row_bounds()
// makes N rows free.
TEST(MpsFormatParserParse, RangesEntryOnNonPrimaryFreeRowIsAcceptedAndIgnored) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "N"}, {5, "FREE1"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "FREE1"}, {50, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "FREE1"}, {25, "5.0"}}) + "\n";
  content += "RANGES\n";
  content += MpsLine({{5, "RNG"}, {15, "FREE1"}, {25, "2.0"}}) + "\n";
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  ParsedModel<double> model;
  EXPECT_NO_THROW(model = parser.parse(WriteTempMps(content)));

  ASSERT_EQ(model.num_rows, 1);
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(model.row_lower(0), -inf);
  EXPECT_EQ(model.row_upper(0), inf);
}

// Same for RANGES.
TEST(MpsFormatParserParse, RangesSecondSetWithDifferentNameStillApplies) {
  std::string content;
  content += "ROWS\n";
  content += MpsLine({{2, "N"}, {5, "COST"}}) + "\n";
  content += MpsLine({{2, "E"}, {5, "R1"}}) + "\n";
  content += MpsLine({{2, "E"}, {5, "R2"}}) + "\n";
  content += "COLUMNS\n";
  content += MpsLine({{5, "X1"}, {15, "COST"}, {25, "1.0"}, {40, "R1"}, {50, "1.0"}}) + "\n";
  content += MpsLine({{5, "X1"}, {15, "R2"}, {25, "1.0"}}) + "\n";
  content += "RHS\n";
  content += MpsLine({{5, "RHS"}, {15, "R1"}, {25, "5.0"}, {40, "R2"}, {50, "5.0"}}) + "\n";
  content += "RANGES\n";
  content += MpsLine({{5, "RNG1"}, {15, "R1"}, {25, "2.0"}}) + "\n";
  content += MpsLine({{5, "RNG2"}, {15, "R2"}, {25, "1.0"}}) + "\n";  // different name
  content += "ENDATA\n";

  MpsFormatParser<double> parser;
  auto model = parser.parse(WriteTempMps(content));
  // R1 (E, rhs=5, range=2): [3,7]
  EXPECT_NEAR(model.row_lower(0), 3.0, kTight);
  EXPECT_NEAR(model.row_upper(0), 7.0, kTight);
  // R2 (E, rhs=5, range=1), from the differently-named RNG2 block: [4,6]
  EXPECT_NEAR(model.row_lower(1), 4.0, kTight);
  EXPECT_NEAR(model.row_upper(1), 6.0, kTight);
}

// ===================== to_kspqp() =====================

TEST(MpsFormatParserToKspqp, SplitsEqualityInequalityAndSkipsFreeRows) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 3;
  model.num_cols = 2;
  model.c.resize(2);
  model.c << 1.0, 1.0;
  model.obj_const = 0.0;

  model.A.resize(3, 2);
  std::vector<Eigen::Triplet<double>> trips = {
      {0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}, {2, 0, 5.0}, {2, 1, 6.0}};
  model.A.setFromTriplets(trips.begin(), trips.end());

  const double inf = std::numeric_limits<double>::infinity();
  model.row_lower.resize(3);
  model.row_upper.resize(3);
  model.row_lower << 5.0, 1.0, -inf;
  model.row_upper << 5.0, 3.0, inf;

  model.col_lower = Eigen::VectorXd::Constant(2, 0.0);
  model.col_upper = Eigen::VectorXd::Constant(2, inf);

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model);

  EXPECT_EQ(pd.n, 2);
  EXPECT_EQ(pd.m, 1);  // row 0 (equality)
  EXPECT_EQ(pd.l, 1);  // row 1 (inequality); row 2 (free) skipped entirely

  EXPECT_NEAR(pd.A.coeff(0, 0), 1.0, kTight);
  EXPECT_NEAR(pd.A.coeff(0, 1), 2.0, kTight);
  EXPECT_NEAR(pd.b(0), 5.0, kTight);

  EXPECT_NEAR(pd.B.coeff(0, 0), 3.0, kTight);
  EXPECT_NEAR(pd.B.coeff(0, 1), 4.0, kTight);
  EXPECT_NEAR(pd.lw(0), 1.0, kTight);
  EXPECT_NEAR(pd.uw(0), 3.0, kTight);
}

// Degenerate eq/ineq splits: all rows equality (l=0, empty B/lw/uw) and all rows
// inequality (m=0, empty A/b).
TEST(MpsFormatParserToKspqp, AllRowsEqualityGivesEmptyInequalityBlock) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 1;
  model.num_cols = 1;
  model.c = Eigen::VectorXd::Zero(1);
  model.A.resize(1, 1);
  model.A.insert(0, 0) = 1.0;
  model.A.makeCompressed();
  model.row_lower.resize(1);
  model.row_upper.resize(1);
  model.row_lower << 5.0;
  model.row_upper << 5.0;  // equality
  model.col_lower = Eigen::VectorXd::Constant(1, 0.0);
  model.col_upper = Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity());

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model);

  EXPECT_EQ(pd.m, 1);
  EXPECT_EQ(pd.l, 0);
  EXPECT_EQ(pd.B.rows(), 0);
  EXPECT_EQ(pd.lw.size(), 0);
  EXPECT_EQ(pd.uw.size(), 0);
}

TEST(MpsFormatParserToKspqp, AllRowsInequalityGivesEmptyEqualityBlock) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 1;
  model.num_cols = 1;
  model.c = Eigen::VectorXd::Zero(1);
  model.A.resize(1, 1);
  model.A.insert(0, 0) = 1.0;
  model.A.makeCompressed();
  model.row_lower.resize(1);
  model.row_upper.resize(1);
  model.row_lower << 1.0;
  model.row_upper << 3.0;  // inequality
  model.col_lower = Eigen::VectorXd::Constant(1, 0.0);
  model.col_upper = Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity());

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model);

  EXPECT_EQ(pd.m, 0);
  EXPECT_EQ(pd.l, 1);
  EXPECT_EQ(pd.A.rows(), 0);
  EXPECT_EQ(pd.b.size(), 0);
}

TEST(MpsFormatParserToKspqp, CapsLargeMagnitudeBoundsToInfinity) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 0;
  model.num_cols = 1;
  model.c.resize(1);
  model.c << 0.0;
  model.A.resize(0, 1);
  model.row_lower.resize(0);
  model.row_upper.resize(0);
  model.col_lower.resize(1);
  model.col_upper.resize(1);
  model.col_lower << -1e30;
  model.col_upper << 1e30;

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model, /*eq_tol=*/1e-12, /*inf_cap=*/1e20);

  EXPECT_EQ(pd.lx(0), -1e20);
  EXPECT_EQ(pd.ux(0), 1e20);
}

TEST(MpsFormatParserToKspqp, SnapsNearEqualBoundsToMidpointWithinTolerance) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 0;
  model.num_cols = 2;
  model.c = Eigen::VectorXd::Zero(2);
  model.A.resize(0, 2);
  model.row_lower.resize(0);
  model.row_upper.resize(0);
  model.col_lower.resize(2);
  model.col_upper.resize(2);
  model.col_lower << 1.999999999, 1.0;  // diff = 2e-9, within eq_tol
  model.col_upper << 2.000000001, 3.0;  // diff = 2.0, outside eq_tol

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model, /*eq_tol=*/1e-8, /*inf_cap=*/std::numeric_limits<double>::infinity());

  EXPECT_EQ(pd.lx(0), pd.ux(0));
  EXPECT_NEAR(pd.lx(0), 2.0, 1e-9);
  EXPECT_NE(pd.lx(1), pd.ux(1));  // outside tolerance: left untouched
  EXPECT_NEAR(pd.lx(1), 1.0, kTight);
  EXPECT_NEAR(pd.ux(1), 3.0, kTight);
}

// The snap condition is `diff <= eq_tol` (inclusive).
TEST(MpsFormatParserToKspqp, SnapBoundaryIsInclusiveAtExactlyEqTol) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 0;
  model.num_cols = 2;
  model.c = Eigen::VectorXd::Zero(2);
  model.A.resize(0, 2);
  model.row_lower.resize(0);
  model.row_upper.resize(0);
  model.col_lower.resize(2);
  model.col_upper.resize(2);

  const double eq_tol = 1e-6;
  model.col_lower << 0.0, 0.0;
  model.col_upper << eq_tol, std::nextafter(eq_tol, 1.0);  // == tol, then just over

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model, eq_tol, std::numeric_limits<double>::infinity());

  EXPECT_EQ(pd.lx(0), pd.ux(0));  // diff == eq_tol exactly: snaps
  EXPECT_NE(pd.lx(1), pd.ux(1));  // diff just over eq_tol: does not snap
}

TEST(MpsFormatParserToKspqp, MaxSenseNegatesCAndQButObjConstIsAlwaysNegated) {
  ParsedModel<double> model;
  model.is_qp = true;
  model.is_min = false;  // MAX
  model.num_rows = 0;
  model.num_cols = 2;
  model.c.resize(2);
  model.c << 1.0, 2.0;
  model.obj_const = 7.0;
  model.A.resize(0, 2);
  model.row_lower.resize(0);
  model.row_upper.resize(0);
  model.col_lower = Eigen::VectorXd::Constant(2, 0.0);
  model.col_upper = Eigen::VectorXd::Constant(2, std::numeric_limits<double>::infinity());

  model.Q.resize(2, 2);
  model.Q.insert(0, 0) = 3.0;
  model.Q.insert(1, 1) = 4.0;
  model.Q.makeCompressed();

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model);

  // obj_const is negated unconditionally as it's read off the RHS section;
  // c and Q are negated only for the MAX -> MIN flip.
  EXPECT_NEAR(pd.obj_const, -7.0, kTight);
  EXPECT_NEAR(pd.c(0), -1.0, kTight);
  EXPECT_NEAR(pd.c(1), -2.0, kTight);
  EXPECT_NEAR(pd.Q.coeff(0, 0), -3.0, kTight);
  EXPECT_NEAR(pd.Q.coeff(1, 1), -4.0, kTight);
}

TEST(MpsFormatParserToKspqp, NonQpModelGetsZeroQOfCorrectSize) {
  ParsedModel<double> model;
  model.is_qp = false;
  model.is_min = true;
  model.num_rows = 0;
  model.num_cols = 3;
  model.c = Eigen::VectorXd::Zero(3);
  model.A.resize(0, 3);
  model.row_lower.resize(0);
  model.row_upper.resize(0);
  model.col_lower = Eigen::VectorXd::Constant(3, 0.0);
  model.col_upper = Eigen::VectorXd::Constant(3, std::numeric_limits<double>::infinity());

  MpsFormatParser<double> parser;
  auto pd = parser.to_kspqp(model);

  EXPECT_EQ(pd.Q.rows(), 3);
  EXPECT_EQ(pd.Q.cols(), 3);
  EXPECT_EQ(pd.Q.nonZeros(), 0);
}
