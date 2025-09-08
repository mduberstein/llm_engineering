
#include <iostream>
#include <iomanip>
#include <chrono>

// Function to perform the calculation
double calculate(unsigned int iterations, int param1, int param2) {
    double result = 1.0;
    for (unsigned int i = 1; i <= iterations; ++i) {
        double j1 = static_cast<double>(i) * param1 - param2;
        result -= (1.0 / j1);
        double j2 = static_cast<double>(i) * param1 + param2;
        result += (1.0 / j2);
    }
    return result;
}

int main() {
    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the calculation with the specified parameters
    double result = calculate(100000000, 4, 1) * 4;

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << "\n";
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}
