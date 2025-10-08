// 1. Сборка с помощью cmake
// cmake -S . -B build
// cmake --build build
// ./build/bmp_show img/geometry.bmp
// 2. Очистка
// rm -S build

// 2. Сборка с помощью g++
// g++ main.cpp -o bmp_show
// ./bmp_show img/geometry.bmp
// rm bmp_show.exe

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <future>
#include <thread>
#include <sstream>

#pragma pack(push, 1)

struct BMPFileHeader {
    uint16_t fileType{};
    uint32_t fileSize{};
    uint16_t reserved1{};
    uint16_t reserved2{};
    uint32_t offsetData{};
};

struct BMPInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression{};
    uint32_t imageSize{};
    int32_t xPixelsPerMeter{};
    int32_t yPixelsPerMeter{};
    uint32_t colorsUsed{};
    uint32_t colorsImportant{};
};

#pragma pack(pop)

class BMPImage {
private:
    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};
    std::vector<uint8_t> pixelData{};
    int rowStride{0};
    constexpr static char WHITE = '#';
    constexpr static char BLACK = ' ';
    constexpr static uint8_t brightness_factor = 128;

    // Получение индекса пикселя
    [[nodiscard]] int getPixelIndex(int x, int y) const {
        return (y * rowStride) + (x * (infoHeader.bitCount / 8));
    }

    // Получение цвета пикселя в формате RGB
    [[nodiscard]] std::string getPixelColor(int x, int y) const {
        int index = getPixelIndex(x, y);
        uint8_t blue = pixelData[index];
        uint8_t green = pixelData[index + 1];
        uint8_t red = pixelData[index + 2];
        
        std::stringstream ss;
        ss << "rgb(" << static_cast<int>(red) << "," 
                  << static_cast<int>(green) << "," 
                  << static_cast<int>(blue) << ")";
        return ss.str();
    }

public:
    void openBMP(const std::string &fileName) {
        std::ifstream file(fileName, std::ios::binary);
        if (!file) {
            throw std::runtime_error("File open error: " + fileName);
        }

        // Чтение заголовков
        file.read(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
        if (file.gcount() != sizeof(fileHeader)) throw std::runtime_error("File header reading error.");

        file.read(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));
        if (file.gcount() != sizeof(infoHeader)) throw std::runtime_error("Info header reading error.");

        if (infoHeader.bitCount != 24 && infoHeader.bitCount != 32) {
            throw std::runtime_error("Unsupported BMP format! Expected 24 or 32 bits.");
        }

        file.seekg(fileHeader.offsetData, std::ios::beg);

        rowStride = (infoHeader.width * (infoHeader.bitCount / 8) + 3) & ~3;
        pixelData.resize(rowStride * infoHeader.height);
        file.read(reinterpret_cast<char *>(pixelData.data()), pixelData.size());
        if (file.gcount() != pixelData.size()) throw std::runtime_error("Pixel read error.");
    }

    [[nodiscard]] bool hasMoreThanTwoColors() const {
        for (int y = 0; y < infoHeader.height; ++y) {
            for (int x = 0; x < infoHeader.width; ++x) {
                int index = getPixelIndex(x, y);
                uint8_t blue = pixelData[index];
                uint8_t green = pixelData[index + 1];
                uint8_t red = pixelData[index + 2];
                if (!(red == 255 && green == 255 && blue == 255) && !(red == 0 && green == 0 && blue == 0))
                    return true;
            }
        }
        return false;
    }

    void convertToBlackAndWhite() {
        auto convertRow = [this](int startRow, int endRow, std::vector<uint8_t> &newPixelData) {
            for (int y = startRow; y < endRow; ++y) {
                for (int x = 0; x < infoHeader.width; ++x) {
                    int index = (y * rowStride) + (x * (infoHeader.bitCount / 8));

                    uint8_t blue = pixelData[index];
                    uint8_t green = pixelData[index + 1];
                    uint8_t red = pixelData[index + 2];

                    double brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue;

                    if (brightness < brightness_factor) {
                        newPixelData[index] = 0;
                        newPixelData[index + 1] = 0;
                        newPixelData[index + 2] = 0;
                    } else {
                        newPixelData[index] = 255;
                        newPixelData[index + 1] = 255;
                        newPixelData[index + 2] = 255;
                    }
                }
            }
        };

        std::vector<uint8_t> newPixelData = pixelData;

        // Получаем максимальное количество потоков
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1; // Если нет доступного количества потоков, то берем 1
        int rowsPerThread = infoHeader.height / numThreads;
        std::vector<std::future<void> > futures;

        for (unsigned int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? infoHeader.height : startRow + rowsPerThread;
            // Последний поток берет оставшиеся строки

            futures.push_back(std::async(std::launch::async, convertRow, startRow, endRow, std::ref(newPixelData)));
        }

        for (auto &future: futures) {
            future.get();
        }

        pixelData = std::move(newPixelData);
    }

    void displayBMP() {
        if (hasMoreThanTwoColors()) {
            std::cout << "Image contains more then 2 colors. Converting to black and white..." << std::endl;
            convertToBlackAndWhite();
        }
        for (int y = infoHeader.height - 1; y >= 0; y -= 2) {
            for (int x = 0; x < infoHeader.width; ++x) {
                int index = getPixelIndex(x, y);
                uint8_t blue = pixelData[index];
                uint8_t green = pixelData[index + 1];
                uint8_t red = pixelData[index + 2];

                std::cout << ((red == 255 && green == 255 && blue == 255) ? WHITE : BLACK);
            }
            std::cout << std::endl;
        }
    }

    void saveSVG(int pixelWidth = 5, int pixelHeight = 5) {
        std::string outputFileName = "output.svg";
        std::ofstream svgFile(outputFileName);
        
        if (!svgFile) {
            throw std::runtime_error("Cannot create SVG file: " + outputFileName);
        }

        // Расчет размеров SVG
        int svgWidth = infoHeader.width * pixelWidth;
        int svgHeight = infoHeader.height * pixelHeight;

        // Заголовок SVG
        svgFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        svgFile << "<svg width=\"" << svgWidth << "\" height=\"" << svgHeight 
                << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
        svgFile << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

        // Создание прямоугольников для каждого пикселя
        for (int y = 0; y < infoHeader.height; ++y) {
            for (int x = 0; x < infoHeader.width; ++x) {
                std::string color = getPixelColor(x, y);
                
                // Пропускаем белые пиксели (фон)
                if (color == "rgb(255,255,255)") {
                    continue;
                }

                int rectX = x * pixelWidth;
                int rectY = (infoHeader.height - 1 - y) * pixelHeight; // Инвертируем Y координату

                svgFile << "  <rect x=\"" << rectX << "\" y=\"" << rectY 
                        << "\" width=\"" << pixelWidth << "\" height=\"" << pixelHeight 
                        << "\" fill=\"" << color << "\"/>\n";
            }
        }

        svgFile << "</svg>\n";
        svgFile.close();

        std::cout << "SVG file saved as: " << outputFileName << std::endl;
        std::cout << "SVG dimensions: " << svgWidth << "x" << svgHeight << " pixels" << std::endl;
    }

    ~BMPImage() {
        pixelData.clear();
    }
};

int main(int argc, char *argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Using: <file-path.bmp>");
        }

        BMPImage image;
        image.openBMP(argv[1]);
        image.displayBMP();
        image.saveSVG(5, 5); // Сохраняем SVG с размерами пикселя 5x5
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}




/*
// 1. Сборка с помощью cmake
// cmake -S . -B build
// cmake --build build
// ./build/bmp_show img/geometry.bmp
// 2. Очистка
// rm -S build

// 2. Сборка с помощью g++
// g++ main.cpp -o bmp_show
// ./bmp_show img/geometry.bmp
// rm bmp_show.exe

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <future>
#include <thread>

#pragma pack(push, 1)

struct BMPFileHeader {
    uint16_t fileType{};
    uint32_t fileSize{};
    uint16_t reserved1{};
    uint16_t reserved2{};
    uint32_t offsetData{};
};

struct BMPInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression{};
    uint32_t imageSize{};
    int32_t xPixelsPerMeter{};
    int32_t yPixelsPerMeter{};
    uint32_t colorsUsed{};
    uint32_t colorsImportant{};
};

#pragma pack(pop)

class BMPImage {
private:
    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};
    std::vector<uint8_t> pixelData{};
    int rowStride{0};
    constexpr static char WHITE = '#';
    constexpr static char BLACK = ' ';
    constexpr static uint8_t brightness_factor = 128;

    // Получение индекса пикселя
    [[nodiscard]] int getPixelIndex(int x, int y) const {
        return (y * rowStride) + (x * (infoHeader.bitCount / 8));
    }

public:
    void openBMP(const std::string &fileName) {
        std::ifstream file(fileName, std::ios::binary);
        if (!file) {
            throw std::runtime_error("File open error: " + fileName);
        }

        // Чтение заголовков
        file.read(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
        if (file.gcount() != sizeof(fileHeader)) throw std::runtime_error("File header reading error.");

        file.read(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));
        if (file.gcount() != sizeof(infoHeader)) throw std::runtime_error("Info header reading error.");

        if (infoHeader.bitCount != 24 && infoHeader.bitCount != 32) {
            throw std::runtime_error("Unsupported BMP format! Expected 24 or 32 bits.");
        }

        file.seekg(fileHeader.offsetData, std::ios::beg);

        rowStride = (infoHeader.width * (infoHeader.bitCount / 8) + 3) & ~3;
        pixelData.resize(rowStride * infoHeader.height);
        file.read(reinterpret_cast<char *>(pixelData.data()), pixelData.size());
        if (file.gcount() != pixelData.size()) throw std::runtime_error("Pixel read error.");
    }

    [[nodiscard]] bool hasMoreThanTwoColors() const {
        for (int y = 0; y < infoHeader.height; ++y) {
            for (int x = 0; x < infoHeader.width; ++x) {
                int index = getPixelIndex(x, y);
                uint8_t blue = pixelData[index];
                uint8_t green = pixelData[index + 1];
                uint8_t red = pixelData[index + 2];
                if (!(red == 255 && green == 255 && blue == 255) && !(red == 0 && green == 0 && blue == 0))
                    return true;
            }
        }
        return false;
    }

    void convertToBlackAndWhite() {
        auto convertRow = [this](int startRow, int endRow, std::vector<uint8_t> &newPixelData) {
            for (int y = startRow; y < endRow; ++y) {
                for (int x = 0; x < infoHeader.width; ++x) {
                    int index = (y * rowStride) + (x * (infoHeader.bitCount / 8));

                    uint8_t blue = pixelData[index];
                    uint8_t green = pixelData[index + 1];
                    uint8_t red = pixelData[index + 2];

                    double brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue;

                    if (brightness < brightness_factor) {
                        newPixelData[index] = 0;
                        newPixelData[index + 1] = 0;
                        newPixelData[index + 2] = 0;
                    } else {
                        newPixelData[index] = 255;
                        newPixelData[index + 1] = 255;
                        newPixelData[index + 2] = 255;
                    }
                }
            }
        };

        std::vector<uint8_t> newPixelData = pixelData;

        // Получаем максимальное количество потоков
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1; // Если нет доступного количества потоков, то берем 1
        int rowsPerThread = infoHeader.height / numThreads;
        std::vector<std::future<void> > futures;

        for (unsigned int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? infoHeader.height : startRow + rowsPerThread;
            // Последний поток берет оставшиеся строки

            futures.push_back(std::async(std::launch::async, convertRow, startRow, endRow, std::ref(newPixelData)));
        }

        for (auto &future: futures) {
            future.get();
        }

        pixelData = std::move(newPixelData);
    }

    void displayBMP() {
        if (hasMoreThanTwoColors()) {
            std::cout << "Image contains more then 2 colors. Converting to black and white..." << std::endl;
            convertToBlackAndWhite();
        }
        for (int y = infoHeader.height - 1; y >= 0; y -= 2) {
            for (int x = 0; x < infoHeader.width; ++x) {
                int index = getPixelIndex(x, y);
                uint8_t blue = pixelData[index];
                uint8_t green = pixelData[index + 1];
                uint8_t red = pixelData[index + 2];

                std::cout << ((red == 255 && green == 255 && blue == 255) ? WHITE : BLACK);
            }
            std::cout << std::endl;
        }
    }

    ~BMPImage() {
        pixelData.clear();
    }
};

int main(int argc, char *argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Using: <file-path.bmp>");
        }

        BMPImage image;
        image.openBMP(argv[1]);
        image.displayBMP();
        image.saveSVG();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
*/