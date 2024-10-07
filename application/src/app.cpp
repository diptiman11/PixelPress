#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;

class Manage {
private:
    string compressionPath;
    string decompressionPath;

public:
    void runCompress() {
        system("python backend/compress.py"); 
    }

    void runDecompress() {
        system("python backend/decompress.py");  
    }

    void actionPerformed(const string& actionCommand) {
        if (actionCommand == "compress") {
            compressionPath = getInput("Enter the path of the file to be compressed: ");
            writeToFile("backend/path.txt", compressionPath);
            runCompress();
        } else if (actionCommand == "decompress") {
            decompressionPath = getInput("Enter the path of the file to be decompressed: ");
            writeToFile("backend/path.txt", decompressionPath);
            runDecompress();
        }
    }

private:
    void writeToFile(const string& filename, const string& content) {
        ofstream writer(filename);
        if (writer.is_open()) {
            writer<<content;
            writer.close();
        } 
        else {
            cerr<<"Unable to open file: "<<filename<<endl;
        }
    }

    string getInput(const string& prompt) {
        string input;
        cout<<prompt;
        getline(cin, input);
        return input;
    }
};

int main() {
    Manage man;
    string actionCommand;

    while (true) {
        cout<<"Enter action (compress/decompress/exit): ";
        getline(cin, actionCommand);

        if(actionCommand=="exit") {
            break;
        }
        man.actionPerformed(actionCommand);
    }
    return 0;
}