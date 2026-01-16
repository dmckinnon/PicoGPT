
    /*
    Auto-generated from: encode.json
    Contains two lookup functions:
        - vocab::vocab_value_for_key(key) -> optional<value>
        - vocab::vocab_key_for_value(value) -> optional<key>
    */

    #include <string>
    #include <unordered_map>
    #include <optional>
    #include <cstdint>


    namespace vocab {

    static const std::unordered_map<std::string, int32_t>& key_to_value() {
        static const std::unordered_map<std::string, int32_t> m = {
        { std::string("\n"), 0 },
    { std::string(" "), 1 },
    { std::string(","), 2 },
    { std::string("."), 3 },
    { std::string("0"), 4 },
    { std::string("1"), 5 },
    { std::string("2"), 6 },
    { std::string("3"), 7 },
    { std::string("4"), 8 },
    { std::string("5"), 9 },
    { std::string("6"), 10 },
    { std::string("?"), 11 },
    { std::string("A"), 12 },
    { std::string("D"), 13 },
    { std::string("H"), 14 },
    { std::string("L"), 15 },
    { std::string("M"), 16 },
    { std::string("N"), 17 },
    { std::string("Q"), 18 },
    { std::string("S"), 19 },
    { std::string("T"), 20 },
    { std::string("U"), 21 },
    { std::string("V"), 22 },
    { std::string("W"), 23 },
    { std::string("X"), 24 },
    { std::string("a"), 25 },
    { std::string("b"), 26 },
    { std::string("c"), 27 },
    { std::string("d"), 28 },
    { std::string("e"), 29 },
    { std::string("f"), 30 },
    { std::string("g"), 31 },
    { std::string("h"), 32 },
    { std::string("i"), 33 },
    { std::string("k"), 34 },
    { std::string("l"), 35 },
    { std::string("m"), 36 },
    { std::string("n"), 37 },
    { std::string("o"), 38 },
    { std::string("p"), 39 },
    { std::string("r"), 40 },
    { std::string("s"), 41 },
    { std::string("t"), 42 },
    { std::string("u"), 43 },
    { std::string("v"), 44 },
    { std::string("w"), 45 },
    { std::string("x"), 46 },
    { std::string("y"), 47 },
    { std::string("z"), 48 }
        };
        return m;
    }

    static const std::unordered_map<int32_t, std::string>& value_to_key() {
        static const std::unordered_map<int32_t, std::string> m = {
        { 0, std::string("\n") },
    { 1, std::string(" ") },
    { 2, std::string(",") },
    { 3, std::string(".") },
    { 4, std::string("0") },
    { 5, std::string("1") },
    { 6, std::string("2") },
    { 7, std::string("3") },
    { 8, std::string("4") },
    { 9, std::string("5") },
    { 10, std::string("6") },
    { 11, std::string("?") },
    { 12, std::string("A") },
    { 13, std::string("D") },
    { 14, std::string("H") },
    { 15, std::string("L") },
    { 16, std::string("M") },
    { 17, std::string("N") },
    { 18, std::string("Q") },
    { 19, std::string("S") },
    { 20, std::string("T") },
    { 21, std::string("U") },
    { 22, std::string("V") },
    { 23, std::string("W") },
    { 24, std::string("X") },
    { 25, std::string("a") },
    { 26, std::string("b") },
    { 27, std::string("c") },
    { 28, std::string("d") },
    { 29, std::string("e") },
    { 30, std::string("f") },
    { 31, std::string("g") },
    { 32, std::string("h") },
    { 33, std::string("i") },
    { 34, std::string("k") },
    { 35, std::string("l") },
    { 36, std::string("m") },
    { 37, std::string("n") },
    { 38, std::string("o") },
    { 39, std::string("p") },
    { 40, std::string("r") },
    { 41, std::string("s") },
    { 42, std::string("t") },
    { 43, std::string("u") },
    { 44, std::string("v") },
    { 45, std::string("w") },
    { 46, std::string("x") },
    { 47, std::string("y") },
    { 48, std::string("z") }
        };
        return m;
    }

    std::optional<int32_t> vocab_value_for_key(const std::string& key) {
        const auto& m = key_to_value();
        auto it = m.find(key);
        if (it == m.end()) return std::nullopt;
        return it->second;
    }

    std::optional<std::string> vocab_key_for_value(int32_t value) {
        const auto& m = value_to_key();
        auto it = m.find(value);
        if (it == m.end()) return std::nullopt;
        return it->second;
    }

    } // namespace vocab
    