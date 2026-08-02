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

    
    struct KeyMaterial {
        double seed;
        double r;
    };

    class KeyDerivation {
    public:
        static KeyMaterial fromPassphrase(const string& passphrase) {
            if (passphrase.empty()) {
                throw invalid_argument("Passphrase must not be empty.");
            }

            uint64_t hSeed = fnv1a(passphrase + "|seed");
            uint64_t hParam = fnv1a(passphrase + "|param");

            double seed = static_cast<double>(hSeed % 1000000ULL) / 1000000.0;
            if (seed <= 0.0 || seed >= 1.0) seed = 0.314159; // avoid fixed points at 0/1

            double r = 3.6 + static_cast<double>(hParam % 400000ULL) / 1000000.0; // [3.6, 4.0)

            return { seed, r };
        }

    private:
        static uint64_t fnv1a(const string& s) {
            uint64_t hash = 14695981039346656037ULL;
            for (unsigned char c : s) {
                hash ^= c;
                hash *= 1099511628211ULL;
            }
            return hash;
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

            // Clamp guards against the (very rare) case where a chaos value
            // rounds up to exactly (i+1), which would index one past the
            // valid range for the Fisher-Yates swap.
            static int clampIndex(int j, int hi) {
                if (j > hi) return hi;
                if (j < 0) return 0;
                return j;
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
                    int j = clampIndex(static_cast<int>(chaos[i] * (i + 1)), i);
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
                    int j = clampIndex(static_cast<int>(chaos[i] * (i + 1)), i);
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

        void printParentPath() const {
            cout << "Last augmenting path (BFS parent pointers): ";
            for (size_t i = 0; i < parent.size(); ++i)
                if (parent[i] != -1)
                    cout << "(" << parent[i] << " -> " << i << ") ";
            cout << "\n";
        }
    };

    class GraphLoader {
    public:
        // Scans the file for the highest node index referenced, without
        // needing a Graph instance yet — lets the caller size the graph
        // correctly before actually loading edges into it.
        static int scanMaxNode(const string& filename) {
            ifstream file(filename);
            if (!file) throw runtime_error("Cannot open graph file.");

            string line;
            int maxNode = -1;
            while (getline(file, line)) {
                istringstream iss(line);
                int u, v, c;
                if (!(iss >> u >> v >> c)) continue;
                maxNode = max({maxNode, u, v});
            }
            return maxNode;
        }

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

    // Bundles everything the report needs, instead of threading five
    // separate out-parameters through simulate().
    struct TransmissionStats {
        int maxFlowCapacity = 0;
        long long totalBytes = 0;
        int rounds = 0;
        double cipherTimeMs = 0.0;
        double simulatedTransferTimeMs = 0.0;
    };

    class TransmissionSimulator {
    private:
        Graph& network;
        int source, sink;

        // Simulated latency per "round" of transmission across the network.
        // One round moves up to maxFlowCapacity bytes; this is what actually
        // ties the max-flow value to a transmission time, rather than the
        // flow computation and the encryption running side by side with no
        // relationship between them (as in the original version).
        static constexpr double TICK_MS = 1.0;

    public:
        TransmissionSimulator(Graph& g, int s, int t) : network(g), source(s), sink(t) {}

        TransmissionStats simulate(Mat& image, Mat& outputEncrypted, const string& passphrase) {
            TransmissionStats stats;

            auto cipherStart = chrono::high_resolution_clock::now();
            KeyMaterial key = KeyDerivation::fromPassphrase(passphrase);
            ChaoticCipher cipher(key.seed, key.r);
            outputEncrypted = image.clone();
            cipher.encrypt(outputEncrypted);
            auto cipherEnd = chrono::high_resolution_clock::now();
            stats.cipherTimeMs = chrono::duration<double, milli>(cipherEnd - cipherStart).count();

            MaxFlow maxflow(network, source, sink);
            stats.maxFlowCapacity = maxflow.run();
            if (stats.maxFlowCapacity <= 0) {
                throw runtime_error("Network has zero max-flow capacity between source and sink; cannot transmit.");
            }

            // Treat the encrypted image as a byte stream and "send" it across
            // the network in chunks bounded by the channel's max-flow
            // capacity. More capacity -> fewer rounds -> less simulated time.
            stats.totalBytes = static_cast<long long>(outputEncrypted.total()) * outputEncrypted.elemSize();
            stats.rounds = static_cast<int>(
                (stats.totalBytes + stats.maxFlowCapacity - 1) / stats.maxFlowCapacity  // ceil division
            );
            stats.simulatedTransferTimeMs = stats.rounds * TICK_MS;

            return stats;
        }

        void decryptImage(Mat& encrypted, Mat& outputDecrypted, const string& passphrase) {
            KeyMaterial key = KeyDerivation::fromPassphrase(passphrase);
            ChaoticCipher cipher(key.seed, key.r);
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
        static void saveReport(const string& filename, double mse, double psnr,
                                const TransmissionStats& stats) {
            ofstream file(filename);
            if (!file) {
                cerr << "Unable to write report file!" << endl;
                return;
            }

            time_t now = time(nullptr);
            file << "Secure Image Transmission Report\n";
            file << "Generated on: " << ctime(&now);
            file << "\n--- Network / Transmission ---\n";
            file << "Max Flow Capacity : " << stats.maxFlowCapacity << "\n";
            file << "Encrypted Size    : " << stats.totalBytes << " bytes\n";
            file << "Rounds to Send    : " << stats.rounds << "\n";
            file << "Simulated Xfer Time: " << stats.simulatedTransferTimeMs << " ms\n";
            file << "\n--- Cipher ---\n";
            file << "Cipher Time (ms)  : " << stats.cipherTimeMs << "\n";
            file << "\n--- Fidelity ---\n";
            file << "MSE        : " << mse << "\n";
            file << "PSNR (dB)  : " << psnr << "\n";
            file << "\nReport saved successfully.\n";
        }
    };

    void interactiveMenu() {
        string inputImage;
        int nodeCount = 6;
        int source = 0, sink = 5;
        string passphrase;
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
                int maxNode = GraphLoader::scanMaxNode(graphPath);
                if (maxNode < 0) {
                    throw runtime_error("Graph file contained no valid edges.");
                }
                nodeCount = maxNode + 1;
            } catch (const exception& e) {
                cerr << "Error reading graph file: " << e.what() << endl;
                return;
            }

            network = Graph(nodeCount);
            GraphLoader::loadFromFile(network, graphPath);

            cout << "Graph loaded successfully! Detected " << nodeCount << " node(s) (0-" << nodeCount - 1 << ").\n";
            cout << "Enter source node [0-" << nodeCount - 1 << "]: ";
            cin >> source;
            cout << "Enter sink node [0-" << nodeCount - 1 << "]: ";
            cin >> sink;

            if (source < 0 || source >= nodeCount || sink < 0 || sink >= nodeCount) {
                cerr << "Error: source/sink out of range for a " << nodeCount << "-node graph.\n";
                return;
            }
        } else {
            cout << "Exiting...\n";
            return;
        }

        cout << "Enter a passphrase (used to derive the cipher key): ";
        cin >> passphrase;

        network.printAdjacencyList();
        network.printCapacityMatrix();

        try {
            Mat original = ImagePreprocessor::loadOriginal(inputImage);

            Mat encrypted, decrypted;
            TransmissionSimulator simulator(network, source, sink);
            TransmissionStats stats = simulator.simulate(original, encrypted, passphrase);
            simulator.decryptImage(encrypted, decrypted, passphrase);

            double mse = simulator.computeMSE(original, decrypted);
            double psnr = simulator.computePSNR(mse);

            string encPath = "out_encrypted.png";
            string decPath = "out_decrypted.png";
            string reportPath = "transmission_report.txt";

            ImagePreprocessor::saveImage(encPath, encrypted);
            ImagePreprocessor::saveImage(decPath, decrypted);
            ReportGenerator::saveReport(reportPath, mse, psnr, stats);

            cout << "\n==== Transmission Summary ====\n";
            cout << "Max flow capacity : " << stats.maxFlowCapacity << "\n";
            cout << "Rounds to send    : " << stats.rounds << "\n";
            cout << "Simulated transfer time: " << stats.simulatedTransferTimeMs << " ms\n";
            cout << "Cipher time       : " << stats.cipherTimeMs << " ms\n";
            cout << "MSE / PSNR        : " << mse << " / " << psnr << " dB\n";

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
