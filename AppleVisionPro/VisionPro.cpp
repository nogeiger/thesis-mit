#include <pybind11/embed.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

namespace py = pybind11;

int main() {
    // Print a message to verify the program started
    std::cout << "Start file execution..." << std::endl;

    // Start the Python interpreter
    py::scoped_interpreter guard{};

    try {
        std::cout << "1" << std::endl;
        // Import the avp_stream library
        py::module avp_stream = py::module::import("avp_stream");

        // Create an instance of VisionProStreamer
        std::string avp_ip = "10.29.194.158";
        py::object VisionProStreamer = avp_stream.attr("VisionProStreamer");
        std::cout << "1.5" << std::endl;
        py::object streamer = VisionProStreamer(avp_ip, true);

        std::cout << "2" << std::endl;
        // Open a text file for writing
        std::ofstream file("streamed_data.txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing.\n";
            return 1;
        }
        std::cout << "3" << std::endl;
        // Timing
        auto start_time = std::chrono::steady_clock::now();
        double timestep = 0.005; // 5 ms
        double next_timestep = timestep;
        std::cout << "4" << std::endl;
        while (true) {
            auto current_time = std::chrono::steady_clock::now();
            double relative_timestamp = std::chrono::duration<double>(current_time - start_time).count();
            std::cout << "5" << std::endl;
            if (relative_timestamp >= next_timestep) {
                // Call the Python method to get the latest data
                py::object latest = streamer.attr("latest");
                py::object right_wrist = latest.attr("get")("right_wrist");

                // Convert right wrist data to string
                std::string right_wrist_str = py::str(right_wrist).cast<std::string>();

                // Write to file
                file << "Timestamp: " << next_timestep << "\n";
                file << "Right Wrist: " << right_wrist_str << "\n\n";
                file.flush();

                // Print live data to the console
                std::cout << "Timestamp: " << next_timestep
                          << " | Right Wrist: " << right_wrist_str << std::endl;

                // Update the next timestep
                next_timestep += timestep;
            }

            // Sleep to reduce CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << "\n";
    }

    return 0;
}
