#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace pico::ui
{

class VerificationReport
{
public:
    struct Row
    {
        std::string label;
        std::string value;
        std::string color;
    };

    VerificationReport(std::string test_name, bool passed, std::string summary_title = "Physics Verification Summary")
            : test_name_(std::move(test_name)), passed_(passed), summary_title_(std::move(summary_title))
    {
    }

    void add_row(std::string label, std::string value, std::string color = YELLOW)
    {
        rows_.push_back({std::move(label), std::move(value), std::move(color)});
    }

    void add_fixed_row(std::string label, double value, int precision = 4, std::string unit = "", std::string color = YELLOW)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        if (!unit.empty())
            ss << " " << unit;
        add_row(std::move(label), ss.str(), std::move(color));
    }

    void add_sci_row(std::string label, double value, int precision = 4, std::string unit = "", std::string color = YELLOW)
    {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(precision) << value;
        if (!unit.empty())
            ss << " " << unit;
        add_row(std::move(label), ss.str(), std::move(color));
    }

    void add_pct_row(std::string label, double pct_value, bool is_ok, int precision = 4)
    {
        add_fixed_row(std::move(label), pct_value, precision, "%", is_ok ? GREEN : RED);
    }

    void print(std::ostream& os = std::cout) const
    {
        os << GRAY << "----------------------------------------------------------------------------------------\n" << RESET;
        if (passed_)
            os << GREEN << BOLD << " ✔ " << test_name_ << " PASSED\n\n" << RESET;
        else
            os << RED << BOLD << " ❌ " << test_name_ << " FAILED\n\n" << RESET;

        os << BOLD << WHITE << " " << summary_title_ << ":\n" << RESET;
        os << GRAY << " ┌──────────────────────────────────────┬──────────────────────────┐\n" << RESET;
        os << " │ " << BOLD << std::left << std::setw(36) << "Metric / Parameter" << RESET << " │ " << BOLD << std::right << std::setw(24) << "Value" << RESET << " │\n";
        os << GRAY << " ├──────────────────────────────────────┼──────────────────────────┤\n" << RESET;

        for (const auto& row : rows_)
        {
            os << " │ " << CYAN << std::left << std::setw(36) << row.label << RESET << " │ " << row.color << std::right << std::setw(24) << row.value << RESET << " │\n";
        }

        os << GRAY << " └──────────────────────────────────────┴──────────────────────────┘\n\n" << RESET;
    }

    bool passed() const
    {
        return passed_;
    }

private:
    std::string      test_name_;
    bool             passed_{false};
    std::string      summary_title_;
    std::vector<Row> rows_;
};

} // namespace pico::ui