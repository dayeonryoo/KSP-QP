#pragma once
#include <cstdlib>
#include <string>

// Minimal `--flag value` command-line parsing shared by the src/*.cpp drivers.
namespace cli {

inline bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

inline std::string get_str(int argc, char** argv, const std::string& flag, const std::string& default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return default_value;
}

inline double get_double(int argc, char** argv, const std::string& flag, double default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::atof(argv[i + 1]);
    }
    return default_value;
}

inline int get_int(int argc, char** argv, const std::string& flag, int default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::atoi(argv[i + 1]);
    }
    return default_value;
}

} // namespace cli
