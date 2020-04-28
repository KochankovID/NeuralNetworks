#ifndef ARTIFICIALNN_MODEL_H
#define ARTIFICIALNN_MODEL_H

#include "ConvolutionLayer.h"
#include "DenceLayer.h"
#include "MaxpoolingLayer.h"
#include "FlattenLayer.h"
#include "vector"
#include "LearnNeuron.h"
#include "Metrics.h"
#include "Vector.h"

namespace NN {

    using std::cout;
    using std::endl;

    template <typename T>
    class Model{
    public:
        Model();

        void add(std::shared_ptr<Layer<T>> layer);

        void learnModel(Ndarray<T> train_data, Ndarray<T> train_out,
                        size_t batch_size, size_t epoches, shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der,
                        const std::vector<shared_ptr<Metr<T>>>& metrics = std::vector<shared_ptr<Metr<T>>>());

        void learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out,
                size_t batch_size, size_t epoches, shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der,
                const std::vector<shared_ptr<Metr<T>>>& metrics = std::vector<shared_ptr<Metr<T>>>());

        void learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, Matrix<Tensor<T>> valitadion_data, Matrix<Tensor<T>> validation_out,
                        size_t batch_size, size_t epoches, shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der,
                        const std::vector<shared_ptr<Metr<T>>>& metrics = std::vector<shared_ptr<Metr<T>>>());

        void learnModel(Ndarray<T> train_data, Ndarray<T> train_out, Ndarray<T> valitadion_data, Ndarray<T> validation_out,
                        size_t batch_size, size_t epoches, shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der,
                        const std::vector<shared_ptr<Metr<T>>>& metrics = std::vector<shared_ptr<Metr<T>>>());

        Tensor<T> predict(const Tensor<T>& x);
        Matrix<Tensor<T>> predict(const Ndarray<T>& x);
        Matrix<Tensor<T>> predict(const Matrix<Tensor<T>>& x);

        void evaluate(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, const std::vector<shared_ptr<Metr<T>>>& metrics);
        void evaluate(Ndarray<T> train_data, Ndarray<T> train_out, const std::vector<shared_ptr<Metr<T>>>& metrics);

        void saveWeight(const std::string file_name = "Weights.txt");
        void getWeight(const std::string file_name = "Weights.txt");


        ~Model() = default;

    private:
        std::vector<std::shared_ptr<Layer<T>>> arr_;
        std::vector<Tensor<T>> TENSOR_IN;
        std::vector<Tensor<T>> TENSOR_OUT;
        std::vector<Tensor<T>> TENSOR_IN_D;
        std::vector<Tensor<T>> TENSOR_OUT_D;


        void showMetrix(size_t ep, size_t bt, size_t koll_of_examples, size_t batch_size,
                        const std::vector<shared_ptr<Metr<T>>> &metrics, const Matrix<T>& metrix_t, size_t base) const;

        void changingWeight(const Matrix<Matrix<T>>& error, shared_ptr<ImpulsGrad<T>> G);

        void errorCalculate(Matrix<Matrix<T>>& error, size_t koll_of_examples, size_t batch_size,  size_t bt);

        void showProgress(size_t koll_of_examples, size_t num_of_examples) const;

    };

    template<typename T>
    Model<T>::Model() {}

    template<typename T>
    void Model<T>::add(std::shared_ptr<Layer<T>> layer) {
        arr_.push_back(layer);
        TENSOR_IN.push_back(Tensor<T>());
        TENSOR_OUT.push_back(Tensor<T>());
        TENSOR_IN_D.push_back(Tensor<T>());
        TENSOR_OUT_D.push_back(Tensor<T>());
    }

    template<typename T>
    void Model<T>::learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, size_t batch_size,
            size_t epoches, shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der, const std::vector<shared_ptr<Metr<T>>> &metrics) {

        if(train_data.getM() != train_out.getM()){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }

        size_t koll_of_examples = train_data.getM();
        Matrix<Matrix<T>> error(1, batch_size);
        Matrix<T> metrix_t(metrics.size(), koll_of_examples);

        cout << "Number of training examples: " << koll_of_examples << endl;

        for(size_t ep = 0; ep < epoches; ep++){
            cout << endl << "epoch: " << ep+1 << '/' << epoches << endl;
            for (size_t bt = 0; bt < koll_of_examples/batch_size; bt++) {

                    for (size_t ex = 0; ex < batch_size; ex++) {
                        predict(train_data[0][bt * batch_size + ex]);

                        error[0][ex] = loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                                          train_out[0][bt * batch_size + ex][0]);

                        for (size_t i = 0; i < metrics.size(); i++) {
                            metrix_t[i][bt * batch_size + ex] = metric_function(*(metrics[i]), TENSOR_OUT.back()[0],
                                                                                train_out[0][bt * batch_size + ex][0]);
                        }
                    }
                cout << std::setw(8) << std::right << bt*batch_size << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, bt*batch_size);
                errorCalculate(error, koll_of_examples, batch_size, bt);
                changingWeight(error, G);
                showMetrix(ep, bt, koll_of_examples, batch_size, metrics, metrix_t, bt*batch_size);
            }

            if((koll_of_examples%batch_size != 0)){
                size_t base = (koll_of_examples-(koll_of_examples % batch_size));
                for(size_t ex = 0; ex < koll_of_examples % batch_size; ex++){
                    predict(train_data[0][base+ex]);

                    error[0][ex] = NN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                            train_out[0][base+ex][0]);

                    for(size_t i = 0; i < metrics.size();i++){
                        metrix_t[i][base+ex] = metric_function(*(metrics[i]), TENSOR_OUT.back()[0],
                                                                        train_out[0][base+ex][0]);
                    }
                }

                cout << std::setw(8) << std::right << base << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, base);
                errorCalculate(error, koll_of_examples, koll_of_examples % batch_size, koll_of_examples/batch_size);
                changingWeight(error, G);
                showMetrix(ep, koll_of_examples/batch_size, koll_of_examples, koll_of_examples % batch_size,
                        metrics, metrix_t, base);
            }
        }
    }

    template<typename T>
    Tensor<T> Model<T>::predict(const Tensor<T>& x) {
        for(size_t i = 0; i < arr_.size(); i++){
            if(i == 0) {
                TENSOR_IN[i] = x;
                TENSOR_OUT[i] = arr_[i]->passThrough(TENSOR_IN[i]);
            }else{
                TENSOR_IN[i] = TENSOR_OUT[i-1];
                TENSOR_OUT[i] = arr_[i]->passThrough(TENSOR_IN[i]);
            }
        }
        return TENSOR_OUT.back();
    }

    template<typename T>
    Matrix<Tensor<T>> Model<T>::predict(const Matrix<Tensor<T>>& x) {
        Matrix<Tensor<T>> result(1, x.getM());
        for(size_t i = 0; i < x.getM(); i++){
            result[0][i] = predict(x[0][i]);
        }
        return result;
    }

    template<typename T>
    void Model<T>::showMetrix(size_t ep, size_t bt, size_t koll_of_examples, size_t batch_size,
                              const std::vector<shared_ptr<Metr<T>>> &metrics, const Matrix<T> &metrix_t, size_t base) const {
        if(ep == 0) {
            for(size_t i = 0; i < metrics.size(); i++) {
                cout << " " << metrics[i]->getName() << ": " << std::setw(6) << std::fixed
                << std::setprecision(4) << std::left << std::setfill('0')
                << mean(metrix_t[i], base + batch_size);
            }
        }else{
            for(size_t i = 0; i < metrics.size(); i++) {
                cout << " " << metrics[i]->getName() << ": " << std::setw(6) << std::fixed
                << std::setprecision(4) << std::left << std::setfill('0')
                << mean(metrix_t[i], koll_of_examples);
            }
        }
        cout << std::setfill(' ');
        cout << std::endl;
    }

    template<typename T>
    void Model<T>::changingWeight(const Matrix<Matrix<T>> &error, shared_ptr<ImpulsGrad<T>> G) {
        for(int i = arr_.size()-1; i >=0; i--){
            if(i == arr_.size()-1){
                TENSOR_IN_D[i] = error[0][0];
                TENSOR_OUT_D[i] = arr_[i]->BackPropagation(TENSOR_IN_D[i], TENSOR_IN[i]);
            }else{
                TENSOR_IN_D[i] = TENSOR_OUT_D[i+1];
                TENSOR_OUT_D[i] = arr_[i]->BackPropagation(TENSOR_IN_D[i], TENSOR_IN[i]);
            }
            arr_[i]->GradDes(*G,TENSOR_IN[i]);
        }
        G->endOfExample();
    }

    template<typename T>
    void
    Model<T>::errorCalculate(Matrix<Matrix<T>> &error, size_t koll_of_examples, size_t batch_size, size_t bt) {
        for (size_t i = 1; i < batch_size; i++) {
            error[0][0] += error[0][i];
        }
        for (size_t i = 0; i < error[0][0].getM(); i++) {
            error[0][0][0][i] /= batch_size;
        }
    }

    template<typename T>
    void Model<T>::showProgress(size_t koll_of_examples, size_t num_of_examples) const {
        double percents = 30 * ((num_of_examples / (double(koll_of_examples) / 100)) * 0.01);
        std::string out = "[";
        for(int i = 0; i < 30; i++){
            if(i == round(percents)){
                out += ">";
            }else{
                if(i < percents) {
                    out += "=";
                }else{
                    out += "-";
                }
            }
        }
        out += "]";
        cout << out;

    }

    template<typename T>
    void Model<T>::saveWeight(std::string file_name) {
        std::ofstream file;
        file.open(file_name);
        if (!file.is_open()) {
            throw DataExeption("Openning file error!");
        }
        for(int i = 0; i < arr_.size(); i++){
            arr_[i]->saveToFile(file);
        }
        file.close();
    }

    template<typename T>
    void Model<T>::getWeight(std::string file_name) {
        std::ifstream file;
        file.open(file_name);
        if (!file.is_open()) {
            throw DataExeption("Openning file error!");
        }
        for(int i = 0; i < arr_.size(); i++){
            arr_[i]->getFromFile(file);
        }
        file.close();
    }

    template<typename T>
    void Model<T>::evaluate(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, 
            const std::vector<shared_ptr<Metr<T>>> &metrics) {
        if(train_data.getM() != train_out.getM()){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }

        size_t koll_of_examples = train_data.getM();
        Matrix<T> metrix_t(metrics.size(), koll_of_examples);

        cout << "Number of training examples: " << koll_of_examples;

        for(size_t ex = 0; ex < koll_of_examples; ex++) {
            predict(train_data[0][ex]);
            for (size_t i = 0; i < metrics.size(); i++) {
                metrix_t[i][ex] = metric_function(*(metrics[i]), TENSOR_OUT.back()[0],
                                                  train_out[0][ex][0]);
            }
        }
        showMetrix(1, 1, koll_of_examples, 1,
                metrics, metrix_t, 0);
    }

    template<typename T>
    void Model<T>::learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, Matrix<Tensor<T>> validation_data,
                              Matrix<Tensor<T>> validation_out, size_t batch_size, size_t epoches,
                              shared_ptr<ImpulsGrad<T>> G, const Metr<T>& loss_func_der,
                              const std::vector<shared_ptr<Metr<T>>> &metrics) {
        if(train_data.getM() != train_out.getM()){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }

        size_t koll_of_examples = train_data.getM();
        Matrix<Matrix<T>> error(1, batch_size);
        Matrix<T> metrix_t(metrics.size(), koll_of_examples);

        cout << "Number of training examples: " << koll_of_examples << endl;

        for(size_t ep = 0; ep < epoches; ep++){
            cout << endl << "epoch: " << ep+1 << '/' << epoches << endl;
            for (size_t bt = 0; bt < koll_of_examples/batch_size; bt++) {

                for (size_t ex = 0; ex < batch_size; ex++) {
                    predict(train_data[0][bt * batch_size + ex]);

                    error[0][ex] = NN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                                      train_out[0][bt * batch_size + ex][0]);

                    for (size_t i = 0; i < metrics.size(); i++) {
                        metrix_t[i][bt * batch_size + ex] = metric_function(*(metrics[i]), TENSOR_OUT.back()[0],
                                                                            train_out[0][bt * batch_size + ex][0]);
                    }
                }
                cout << std::setw(8) << std::right << bt*batch_size << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, bt*batch_size);
                errorCalculate(error, koll_of_examples, batch_size, bt);
                changingWeight(error, G);
                showMetrix(ep, bt, koll_of_examples, batch_size, metrics, metrix_t, bt*batch_size);
            }

            if((koll_of_examples%batch_size != 0)){
                size_t base = (koll_of_examples-(koll_of_examples % batch_size));
                for(size_t ex = 0; ex < koll_of_examples % batch_size; ex++){
                    predict(train_data[0][base+ex]);

                    error[0][ex] = NN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                                      train_out[0][base+ex][0]);

                    for(size_t i = 0; i < metrics.size();i++){
                        metrix_t[i][base+ex] = metric_function(*(metrics[i]), TENSOR_OUT.back()[0],
                                                               train_out[0][base+ex][0]);
                    }
                }

                cout << std::setw(8) << std::right << base << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, base);
                errorCalculate(error, koll_of_examples, koll_of_examples % batch_size, koll_of_examples/batch_size);
                changingWeight(error, G);
                showMetrix(ep, koll_of_examples/batch_size, koll_of_examples, koll_of_examples % batch_size,
                           metrics, metrix_t, base);
            }
            cout << "Validation: ";
            evaluate(validation_data, validation_out, metrics);
        }
    }

    template<typename T>
    void Model<T>::learnModel(Ndarray<T> train_data, Ndarray<T>  train_out, size_t batch_size, size_t epoches,
                              shared_ptr<ImpulsGrad<T>> G, const Metr<T> &loss_func_der,
                              const vector<shared_ptr<Metr<T>>> &metrics) {
        if(train_data.shape().size() < 2) {
            throw std::runtime_error("Wrong shape!");
        }
        if(train_data.shape()[0] != train_out.shape()[0]){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }
        Vector<Tensor<T>> train_data_temp(train_data.shape()[0]);
        Vector<Tensor<T>> train_out_temp(train_out.shape()[0]);
        for(int i =0; i < train_out.shape()[0]; i++){
            train_data_temp[i] = Tensor<T>(train_data.subArray(1,i));
            train_out_temp[i] = Tensor<T>(train_out.subArray(1,i));
        }
        learnModel(train_data_temp, train_out_temp, batch_size, epoches, G, loss_func_der, metrics);
    }

    template<typename T>
    void
    Model<T>::evaluate(Ndarray<T> train_data, Ndarray<T> train_out, const vector<shared_ptr<Metr<T>>> &metrics) {
        if(train_data.shape().size() < 2) {
            throw std::runtime_error("Wrong shape!");
        }
        if(train_data.shape()[0] != train_out.shape()[0]){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }
        Vector<Tensor<T>> train_data_temp(train_data.shape()[0]);
        Vector<Tensor<T>> train_out_temp(train_out.shape()[0]);
        for(int i =0; i < train_out.shape()[0]; i++){
            train_data_temp[i] = Tensor<T>(train_data.subArray(1,i));
            train_out_temp[i] = Tensor<T>(train_out.subArray(1,i));
        }
        evaluate(train_data_temp, train_out_temp, metrics);
    }

    template<typename T>
    void Model<T>::learnModel(Ndarray<T> train_data, Ndarray<T> train_out, Ndarray<T> valitadion_data,
                              Ndarray<T> validation_out, size_t batch_size, size_t epoches,
                              shared_ptr<ImpulsGrad<T>> G, const Metr<T> &loss_func_der,
                              const vector<shared_ptr<Metr<T>>> &metrics) {
        if(train_data.shape().size() < 2) {
            throw std::runtime_error("Wrong shape!");
        }
        if(train_data.shape()[0] != train_out.shape()[0]){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }
        Vector<Tensor<T>> train_data_temp(train_data.shape()[0]);
        Vector<Tensor<T>> train_out_temp(train_out.shape()[0]);
        for(int i =0; i < train_out.shape()[0]; i++){
            train_data_temp[i] = Tensor<T>(train_data.subArray(1,i));
            train_out_temp[i] = Tensor<T>(train_out.subArray(1,i));
        }

        if(valitadion_data.shape().size() < 2) {
            throw std::runtime_error("Wrong shape!");
        }
        if(valitadion_data.shape()[0] != validation_out.shape()[0]){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }
        Vector<Tensor<T>> valitadion_data_temp(train_data.shape()[0]);
        Vector<Tensor<T>> valitadion_out_temp(train_out.shape()[0]);
        for(int i =0; i < train_out.shape()[0]; i++){
            valitadion_data_temp[i] = Tensor<T>(train_data.subArray(1,i));
            valitadion_out_temp[i] = Tensor<T>(train_out.subArray(1,i));
        }
        learnModel(train_data, train_out, valitadion_data, validation_out, batch_size, epoches, G, loss_func_der, metrics);
    }

    template<typename T>
    Matrix<Tensor<T>> Model<T>::predict(const Ndarray<T> &x) {
        if(x.shape().size() < 2) {
            throw std::runtime_error("Wrong shape!");
        }
        Matrix<Tensor<T>> train_data_temp(x.shape()[0]);
        for(int i =0; i < x.shape()[0]; i++){
            train_data_temp[i] = Tensor<T>(x.subArray(1,i));
        }
        predict(train_data_temp);
    }

#define D_Model Model<double>
#define F_Model Model<float>
#define I_Model Model<int>
}

#endif //ARTIFICIALNN_MODEL_H
