#ifndef ARTIFICIALNN_DATA_H
#define ARTIFICIALNN_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include "Neyrons.h"
#include <random>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

namespace ANN {

    using cv::Mat;
    using cv::imread;
    using boost::filesystem::path;
    using boost::filesystem::directory_entry;
    using boost::filesystem::directory_iterator;
    using boost::filesystem::filesystem_error;
    using std::cout;
    using std::endl;

    template<typename T>
    void getDataFromTextFile(ANN::Matrix<ANN::Matrix<T>> &input, const std::string &str);

    template<typename T>
    void saveWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str);

    template<typename T>
    void saveWeightsTextFile(ANN::Matrix<ANN::Neyron<T>> &neyron, const std::string &str);

    template<typename T>
    void getWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str);

    template<typename T>
    void getWeightsTextFile(ANN::Matrix<ANN::Neyron<T> > &neyron, const std::string &str);

    template<typename T>
    void saveFiltersTextFile(ANN::Filter<T> &filter, const std::string &str);

    template<typename T>
    void saveFiltersTextFile(ANN::Matrix<ANN::Filter<T>> &filter, const std::string &str);

    template<typename T>
    void getFiltersTextFile(ANN::Filter<T> &filter, const std::string &str);

    template<typename T>
    void getFiltresTextFile(ANN::Matrix<ANN::Filter<T> > &filter, const std::string &str);

    template<typename T>
    std::pair<Matrix<Tensor<T>>, Matrix<Tensor<T>>> getImageDataFromDirectory(const std::string &dir_path,
            int imread_mode, double k, bool shuf = true);

// Класс исключения ------------------------------------------------------
    class DataExeption : public std::logic_error {
    public:
        DataExeption(std::string str) : std::logic_error(str) {};

        ~DataExeption() {};
    };

    template<typename T>
    void getDataFromTextFile(ANN::Matrix<ANN::Matrix<T> > &input, const std::string &str) {
        std::ifstream TeachChoose;
        TeachChoose.open(str);
        if (!TeachChoose.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < input.getN(); i++) {
            for (int j = 0; j < input.getM(); j++) {
                TeachChoose >> input[i][j];
            }
        }
        TeachChoose.close();
    }

    template<typename T>
    void saveWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str) {
        std::ofstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fWeights << neyron;
        fWeights.close();
    }

    template<typename T>
    void saveWeightsTextFile(ANN::Matrix<ANN::Neyron<T>> &neyron, const std::string &str) {
        std::ofstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                fWeights << neyron[i][j];
            }
        }
        fWeights.close();
    }

    template<typename T>
    void getWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str) {
        std::ifstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fWeights >> neyron;
        fWeights.close();
    }

    template<typename T>
    void getWeightsTextFile(ANN::Matrix<ANN::Neyron<T> > &neyron, const std::string &str) {
        std::ifstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                fWeights >> neyron[i][j];
            }
        }
        fWeights.close();
    }

    template<typename T>
    void saveFiltersTextFile(Filter <T> &filter, const std::string &str) {
        std::ofstream fFilters;
        fFilters.open(str);
        if (!fFilters.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fFilters << filter;
        fFilters.close();
    }

    template<typename T>
    void ANN::saveFiltersTextFile(Matrix <ANN::Filter<T>> &filter, const std::string &str) {
        std::ofstream fFilters;
        fFilters.open(str);
        if (!fFilters.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < filter.getN(); i++) {
            for (int j = 0; j < filter.getM(); j++) {
                fFilters << filter[i][j];
            }
        }
        fFilters.close();
    }

    template<typename T>
    void getFiltersTextFile(Filter <T> &filter, const std::string &str) {
        std::ifstream fFilters;
        fFilters.open(str);
        if (!fFilters.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fFilters >> filter;
        fFilters.close();
    }

    template<typename T>
    void getFiltresTextFile(Matrix <ANN::Filter<T>> &filter, const std::string &str) {
        std::ifstream fFilters;
        fFilters.open(str);
        if (!fFilters.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < filter.getN(); i++) {
            for (int j = 0; j < filter.getM(); j++) {
                fFilters >> filter[i][j];
            }
        }
        fFilters.close();
    }

    template<typename T>
    std::pair<Matrix<Tensor<T>>, Matrix<Tensor<T>>> getImageDataFromDirectory(const std::string &dir_path,
            int imread_mode, double k, bool shuf) {
        Matrix<Tensor<T>> data_x;
        Matrix<Tensor<T>> data_y;
        size_t number_of_examples = 0;
        path p(dir_path);
        try
        {
            if (exists(p))
            {
                if (is_directory(p))
                {
                    std::vector<path> v;

                    for (auto&& x : directory_iterator(p))
                        v.push_back(x.path());

                    std::sort(v.begin(), v.end());

                    for(int i = 0; i < v.size(); i++)
                        for(auto&& x : directory_iterator(v[i]))
                            number_of_examples++;

                    cout << "Found " << number_of_examples << " images belonging to "
                        << v.size() << " classes" << endl;

                    data_x = Matrix<Tensor<T>>(1, number_of_examples);
                    data_y = Matrix<Tensor<T>>(1, number_of_examples);

                    size_t base = 0;

                    for(int i = 0; i < v.size(); i++){
                        size_t cur_class_numer_of_examplex = 0;

                        for (auto&& x : directory_iterator(v[i]))
                            cur_class_numer_of_examplex++;
                        int counter = 0;

                        for (auto&& x : directory_iterator(v[i])){
                            if(is_directory(x)){
                                throw std::runtime_error("Directory in the dataset!");
                            }

                            Mat image = imread(x.path().string(), imread_mode);
                            Tensor<T> temp = Tensor<T>(image.rows, image.cols, image.channels());
                            for(size_t ii = 0; ii < image.channels(); ii++){
                                temp[ii] = Matrix<T>(image.rows, image.cols);

                                for(size_t xx = 0; xx < image.rows; xx++){
                                    for(size_t yy = 0; yy < image.cols; yy++){
                                        temp[ii][xx][yy] = (T)image.at<uchar>(xx, yy) * k;
                                    }
                                }
                            }
                            data_x[0][base+counter] = temp;
                            data_y[0][base+counter] = Tensor<T>(1, v.size(), 1);
                            data_y[0][base+counter].Fill(0);
                            data_y[0][base+counter++][0][0][i] = 1;
                        }
                        base += cur_class_numer_of_examplex;

                    }
                }
                else
                    throw std::logic_error("Path exists, but is not a regular file or directory\n");
            }
            else
                throw std::logic_error("Path does not exist!\n");
        }
        catch (const filesystem_error& ex)
        {
            cout << ex.what() << '\n';
        }

        if(shuf){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(data_x[0], data_x[0] + number_of_examples, std::default_random_engine(seed));
            std::shuffle(data_y[0], data_y[0] + number_of_examples, std::default_random_engine(seed));
        }
        return std::make_pair(data_x, data_y);
    }
}
#endif //ARTIFICIALNN_DATA_H
