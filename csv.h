#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>


class CSVHandler {
private:
    std::string filename;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data;
    std::vector<std::string> headers;
public:
    CSVHandler(const std::string& filename) : filename(filename) {
        readCSV();
    }

    void readCSV() {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::ofstream create(filename);
            create.close();
            return;
        }

        std::string line;
        bool isFirstLine = true;
        while (std::getline(file, line)) {
            std::istringstream lineStream(line);
            std::string cell;
            std::string rowKey;
            std::getline(lineStream, rowKey, ',');
            if (isFirstLine) {
                // Skip the first column of the first row
                isFirstLine = false;
                while (std::getline(lineStream, cell, ',')) {
                    headers.push_back(cell);
                }
            }
            else {
                int colIndex = 0;
                while (std::getline(lineStream, cell, ',')) {
                    data[rowKey][headers[colIndex]] = cell;
                    colIndex++;
                }
            }
        }
        file.close();
    }

    std::string getValue(const std::string& rowKey, const std::string& colKey) const {
        if (data.find(rowKey) != data.end() && data.at(rowKey).find(colKey) != data.at(rowKey).end()) {
            return data.at(rowKey).at(colKey);
        }
        return "";
    }

    void setValue(const std::string& rowKey, const std::string& colKey, const std::string& value) {
        // もしすでにデータが入っている場合に，エラーを返す
        if (data[rowKey].find(colKey) != data[rowKey].end()) {
            std::cout << "[csv.h | setValue] Error: Attempt to overwrite existing data." << std::endl;  // 警告は必ず出すが，exitするかはユーザ設定による
            if (CSV_OVERRIDE_STOP)  throw std::runtime_error("Error: Attempt to overwrite existing data.");
        }
        data[rowKey][colKey] = value;
        if (std::find(headers.begin(), headers.end(), colKey) == headers.end()) {
            headers.push_back(colKey);
        }
    }

    void saveChanges() const {
        std::ofstream file(filename);
        // Write headers
        file << ",";
        for (size_t i = 0; i < headers.size(); ++i) {
            if (i > 0) file << ",";
            file << headers[i];
        }
        file << "\n";

        // Write data
        for (const auto& pair : data) {
            file << pair.first;
            for (size_t i = 0; i < headers.size(); ++i) {
                file << ",";
                if (pair.second.find(headers[i]) != pair.second.end()) {
                    file << pair.second.at(headers[i]);
                }
            }
            file << "\n";
        }
        file.close();
    }
};