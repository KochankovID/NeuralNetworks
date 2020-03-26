#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace cv;
using namespace std;

int main()
{
	string folder;
	Mat image;
	Mat image_2;
	string file;
	string path;
	string path_2;
	for (int i = 0; i < 10; i++) {
		ofstream out_2("./resources/"+to_string(i)+"_tests.txt");
        ofstream out("./resources/"+to_string(i)+".txt");
		string folder = "../../mnist_png/training/" + to_string(i) + "/";
		string folder_2 = "../../mnist_png/testing/" + to_string(i) + "/";
		for (int j = 1; j < 5300; j++) {
			file = " (" + to_string(j) + ").png";
			path = folder + file;
			path_2 = folder_2 + file;
			image = imread(path, IMREAD_GRAYSCALE);
			image_2 = imread(path_2, IMREAD_GRAYSCALE);

			out << image.rows << ' ' << image.cols << endl;
			for (int i = 0; i < image.rows; i++) {
				for (int j = 0; j < image.cols; j++) {
//					if ((int)image.at<uchar>(i, j) == 0) {
//						out << -1 << ' ';
//					}
//					else {
//						out << 1 << ' ';
//					}
                out << (double)image.at<uchar>(i, j) / 255<< ' ';
				}
				out << endl;
			}

            out_2 << image_2.rows << ' ' << image_2.cols << endl;
            for (int i = 0; i < image_2.rows; i++) {
                for (int j = 0; j < image_2.cols; j++) {
//                    if ((int)image_2.at<uchar>(i, j) == 0) {
//                        out_2 << -1 << ' ';
//                    }
//                    else {
//                        out_2 << 1 << ' ';
//                    }
                    out_2 << (double)image.at<uchar>(i, j) / 255<< ' ';
                }
                out_2 << endl;
            }
		}
	}
	return 0;
}
