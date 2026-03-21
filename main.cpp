// ===== Secure Image Transmission - Unified Headers =====
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <set>
#include <map>
#include <filesystem>


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;


namespace SecureImage {

    class ImagePreprocessor {
    public:
        static Mat loadOriginal(const string& path) {
            Mat img = imread(path, IMREAD_COLOR);
            if (img.empty()) {
                throw runtime_error("Failed to load image: " + path);
            }

            return img;
        }

        static void saveImage(const string& path, const Mat& image) {
            imwrite(path, image);
        }
    };

    class ChaoticCipher {
        private:
            double x0;   // seed
            double r;    // control parameter

            vector<double> generateSequence(int size) const {
                vector<double> seq(size);
                double x = x0;

                for (int i = 0; i < size; i++) {
                    x = r * x * (1 - x);
                    seq[i] = x;
                }
                return seq;
            }

        public:
            ChaoticCipher(double seed, double param) : x0(seed), r(param) {}

            void encrypt(Mat& img) const {
                int total = img.rows * img.cols;

                Vec3b* ptr = img.ptr<Vec3b>(0);

                vector<Vec3b> pixels(total);
                for (int i = 0; i < total; i++)
                    pixels[i] = ptr[i];

                vector<double> chaos = generateSequence(total);

                // ===== Fisher–Yates Permutation =====
                vector<int> perm(total);
                iota(perm.begin(), perm.end(), 0);

                for (int i = total - 1; i > 0; i--) {
                    int j = static_cast<int>(chaos[i] * (i + 1));
                    swap(perm[i], perm[j]);
                }

                // Apply permutation
                vector<Vec3b> shuffled(total);
                for (int i = 0; i < total; i++) {
                    shuffled[i] = pixels[perm[i]];
                }

                // ===== Diffusion =====
                for (int i = 0; i < total; i++) {
                    int key = static_cast<int>((chaos[i] * 1e6 + i * 31)) % 256;

                    for (int c = 0; c < 3; c++) {
                        shuffled[i][c] ^= key;
                    }
                }

                // ===== Chaining =====
                for (int i = 1; i < total; i++) {
                    for (int c = 0; c < 3; c++) {
                        shuffled[i][c] ^= shuffled[i - 1][c];
                   }
                }

                // Write back
                for (int i = 0; i < total; i++)
                    ptr[i] = shuffled[i];
            }

            void decrypt(Mat& img) const {
                int total = img.rows * img.cols;

                Vec3b* ptr = img.ptr<Vec3b>(0);

                vector<Vec3b> pixels(total);
                for (int i = 0; i < total; i++)
                    pixels[i] = ptr[i];

                // Regenerate Chaos
                vector<double> chaos = generateSequence(total);

                // ===== Recreate permutation =====
                vector<int> perm(total);
                iota(perm.begin(), perm.end(), 0);

                for (int i = total - 1; i > 0; i--) {
                    int j = static_cast<int>(chaos[i] * (i + 1));
                    swap(perm[i], perm[j]);
                }

                // ===== Reverse chaining =====
                for (int i = total - 1; i > 0; i--) {
                    for (int c = 0; c < 3; c++) {
                        pixels[i][c] ^= pixels[i - 1][c];
                    }
                }

                // ===== Reverse diffusion =====
                for (int i = 0; i < total; i++) {
                    int key = static_cast<int>((chaos[i] * 1e6 + i * 31)) % 256;

                    for (int c = 0; c < 3; c++) {
                        pixels[i][c] ^= key;
                    }
                }

                // ===== Reverse permutation =====
                vector<Vec3b> original(total);
                for (int i = 0; i < total; i++) {
                    original[perm[i]] = pixels[i];
                }

                for (int i = 0; i < total; i++)
                    ptr[i] = original[i];
            }
        };

    class FileManager {
    public:
        static vector<string> listFiles(const string& directory) {
            vector<string> files;
            for (const auto& entry : filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file())
                    files.push_back(entry.path().string());
            }
            return files;
        }

        static void copyFile(const string& src, const string& dst) {
            filesystem::copy(src, dst, filesystem::copy_options::overwrite_existing);
        }
    };
}



namespace SecureImage {

    class Graph {
    private:
        int numNodes;
        vector<vector<int>> capacity;
        vector<vector<int>> flow;
        vector<vector<int>> adj;

    public:
        Graph(int n) : numNodes(n),
                       capacity(n, vector<int>(n, 0)),
                       flow(n, vector<int>(n, 0)),
                       adj(n) {}

        void addEdge(int u, int v, int cap) {
            if (u >= numNodes || v >= numNodes) return;
            capacity[u][v] = cap;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        void resetFlow() {
            for (int i = 0; i < numNodes; ++i)
                for (int j = 0; j < numNodes; ++j)
                    flow[i][j] = 0;
        }

        const vector<int>& getNeighbors(int u) const {
            return adj[u];
        }

        int getCapacity(int u, int v) const {
            return capacity[u][v];
        }

        void updateFlow(int u, int v, int delta) {
            flow[u][v] += delta;
            flow[v][u] -= delta;
            capacity[u][v] -= delta;
            capacity[v][u] += delta;
        }

        void printCapacityMatrix() const {
            cout << "\nCapacity Matrix:\n";
            for (int i = 0; i < numNodes; ++i) {
                for (int j = 0; j < numNodes; ++j)
                    cout << setw(3) << capacity[i][j] << " ";
                cout << "\n";
            }
        }

        void printAdjacencyList() const {
            cout << "\nAdjacency List:\n";
            for (int i = 0; i < numNodes; ++i) {
                cout << i << ": ";
                for (int j : adj[i])
                    cout << j << " ";
                cout << "\n";
            }
        }

        int size() const {
            return numNodes;
        }
    };

    class MaxFlow {
    private:
        Graph& graph;
        vector<int> parent;
        int source, sink;

        bool bfs() {
            int n = graph.size();
            parent.assign(n, -1);
            vector<bool> visited(n, false);
            queue<int> q;
            q.push(source);
            visited[source] = true;

            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : graph.getNeighbors(u)) {
                    if (!visited[v] && graph.getCapacity(u, v) > 0) {
                        parent[v] = u;
                        visited[v] = true;
                        if (v == sink) return true;
                        q.push(v);
                    }
                }
            }
            return false;
        }

    public:
        MaxFlow(Graph& g, int s, int t) : graph(g), source(s), sink(t) {}

        int run() {
            int maxFlow = 0;
            while (bfs()) {
                int pathFlow = numeric_limits<int>::max();
                for (int v = sink; v != source; v = parent[v]) {
                    int u = parent[v];
                    pathFlow = min(pathFlow, graph.getCapacity(u, v));
                }

                for (int v = sink; v != source; v = parent[v]) {
                    int u = parent[v];
                    graph.updateFlow(u, v, pathFlow);
                }

                maxFlow += pathFlow;
            }
            return maxFlow;
        }

        void printParentPath() {
            cout << "Augmenting Path: ";
            for (int i = 0; i < parent.size(); ++i)
                if (parent[i] != -1)
                    cout << "(" << parent[i] << " -> " << i << ") ";
            cout << "\n";
        }
    };

    class GraphLoader {
    public:
        static void loadFromFile(Graph& g, const string& filename) {
            ifstream file(filename);
            if (!file) throw runtime_error("Cannot open graph file.");

            string line;
            while (getline(file, line)) {
                istringstream iss(line);
                int u, v, c;
                if (!(iss >> u >> v >> c)) continue;
                g.addEdge(u, v, c);
            }
        }
    };

} 


namespace SecureImage {

    class TransmissionSimulator {
    private:
        Graph& network;
        int source, sink;

    public:
        TransmissionSimulator(Graph& g, int s, int t) : network(g), source(s), sink(t) {}

        int simulate(Mat& image, Mat& outputEncrypted, int key, double& timeTaken) {
            auto start = chrono::high_resolution_clock::now();

            double seed = (key % 1000) / 1000.0;
            if (seed <= 0.0) seed = 0.5;

            ChaoticCipher cipher(seed, 3.99);
            outputEncrypted = image.clone();
            cipher.encrypt(outputEncrypted);

            MaxFlow maxflow(network, source, sink);
            int maxFlow = maxflow.run();

            auto end = chrono::high_resolution_clock::now();
            timeTaken = chrono::duration<double, milli>(end - start).count();

            return maxFlow;
        }

        void decryptImage(Mat& encrypted, Mat& outputDecrypted, int key) {
            double seed = (key % 1000) / 1000.0;
            if (seed <= 0.0) seed = 0.5;

            ChaoticCipher cipher(seed, 3.99);
            outputDecrypted = encrypted.clone();
            cipher.decrypt(outputDecrypted);
        }

        double computeMSE(const Mat& img1, const Mat& img2) {
            Mat diff;
            absdiff(img1, img2, diff);
            diff.convertTo(diff, CV_32F);
            diff = diff.mul(diff);
            Scalar sumSq = sum(diff);
            double mse = (sumSq[0] + sumSq[1] + sumSq[2]) / (img1.total() * 3.0);
            return mse;
        }

        double computePSNR(double mse) {
            if (mse == 0) return INFINITY;
            return 10.0 * log10((255 * 255) / mse);
        }

        void compareAndDisplay(const Mat& original, const Mat& decrypted) {
            double mse = computeMSE(original, decrypted);
            double psnr = computePSNR(mse);

            cout << "\n==== Performance Metrics ====" << endl;
            cout << "MSE  : " << mse << endl;
            cout << "PSNR : " << psnr << " dB" << endl;

            namedWindow("Original", WINDOW_NORMAL);
            namedWindow("Decrypted", WINDOW_NORMAL);
            imshow("Original", original);
            imshow("Decrypted", decrypted);
            waitKey(0);
            destroyAllWindows();
        }
    };
}



namespace SecureImage {

    class ReportGenerator {
    public:
        static void saveReport(const string& filename, double mse, double psnr, int flow, double timeMs) {
            ofstream file(filename);
            if (!file) {
                cerr << "Unable to write report file!" << endl;
                return;
            }

            time_t now = time(nullptr);
            file << "Secure Image Transmission Report\n";
            file << "Generated on: " << ctime(&now);
            file << "\n--- Metrics ---\n";
            file << "Max Flow   : " << flow << "\n";
            file << "Time (ms)  : " << timeMs << "\n";
            file << "MSE        : " << mse << "\n";
            file << "PSNR (dB)  : " << psnr << "\n";
            file << "\nReport saved successfully.\n";
        }
    };

    void interactiveMenu() {
        string inputImage;
        int nodeCount = 6;
        int source = 0, sink = 5;
        uchar key = 0xB4;
        int choice;

        cout << "\n====== Secure Image Transmission App ======\n";
        cout << "1. Use default graph\n";
        cout << "2. Load graph from file\n";
        cout << "3. Exit\n";
        cout << "==========================================\n";
        cout << "Enter choice: ";
        cin >> choice;

        Graph network(nodeCount);

        if (choice == 1) {
            inputImage = "sample1.jpg";  // default image
            network.addEdge(0, 1, 15);
            network.addEdge(0, 2, 10);
            network.addEdge(1, 3, 10);
            network.addEdge(2, 4, 15);
            network.addEdge(3, 5, 10);
            network.addEdge(4, 5, 10);
        } else if (choice == 2) {
            cout << "Enter path to image file (e.g., E:\\sample2.jpg): ";
            cin >> inputImage;
        
            string graphPath;
            cout << "Enter path to graph file (e.g., E:\\graph2.txt): ";
            cin >> graphPath;
        
            try {
                GraphLoader::loadFromFile(network, graphPath);
                cout << "Graph loaded successfully!\n";
            } catch (const exception& e) {
                cerr << "Error: " << e.what() << endl;
                return;
            }
        } else {
            cout << "Exiting...\n";
            return;
        }

        try {
            Mat original = ImagePreprocessor::loadOriginal(inputImage);

            Mat encrypted, decrypted;
            double elapsed = 0;
            TransmissionSimulator simulator(network, source, sink);
            int maxFlow = simulator.simulate(original, encrypted, key, elapsed);
            simulator.decryptImage(encrypted, decrypted, key);

            double mse = simulator.computeMSE(original, decrypted);
            double psnr = simulator.computePSNR(mse);

            string encPath = "out_encrypted.png";
            string decPath = "out_decrypted.png";
            string reportPath = "transmission_report.txt";

            ImagePreprocessor::saveImage(encPath, encrypted);
            ImagePreprocessor::saveImage(decPath, decrypted);
            ReportGenerator::saveReport(reportPath, mse, psnr, maxFlow, elapsed);

            cout << "\nTransmission complete.\n";
            cout << "Encrypted image saved to: " << encPath << "\n";
            cout << "Decrypted image saved to: " << decPath << "\n";
            cout << "Report saved to: " << reportPath << "\n";

        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
        }
    }

} 

int main(int argc, char* argv[]) {
    SecureImage::interactiveMenu();
    return 0;
}
