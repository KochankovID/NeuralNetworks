#include "gtest/gtest.h"
#define TEST_Ndarray
#include "Ndarray.h"

using namespace NN;

class Ndarray_Methods : public ::testing::Test {
public:
    Ndarray_Methods(): B({3,3,3}), A({2,2}) {}

    ~Ndarray_Methods() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        for(int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    B.buffer[k*6+i*3+j] = 1;
                }
            }
        }
        A({0,0}) = 5;
        A({0, 1})= 10;
        A({1,0}) = 2;
        A({1,1}) = 11;
    }
    void TearDown() { /* called after every test */ }
public:
    Ndarray<double> B;
    Ndarray<double> A;
};
class Ndarray_Methods_Turple : public ::testing::TestWithParam<std::tuple<size_t , size_t, size_t>> {
public:
    Ndarray_Methods_Turple(): B({3,3,3}) {}

    ~Ndarray_Methods_Turple() { /* free protected members here */ }

    void SetUp() {
        /* called before every test */
        for(int k = 0; k < 3; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    B.buffer[k*9+i*3+j] = 1;
                }
            }
        }
    }
    void TearDown() { /* called after every test */ }
public:
    Ndarray<double> B;
};

TEST(Ndarray_constructors, default_constructor_works){
    // Arrange

    // Act

    // Assert
    EXPECT_NO_THROW(Ndarray<int> ndarray);
}

TEST(Ndarray_constructors, default_constructor){
    // Arrange
    Ndarray<int> ndarray;

    // Act

    // Assert
    EXPECT_EQ(ndarray.shape_[0], 0);
    EXPECT_EQ(ndarray.shape_.size(), 1);
    EXPECT_EQ(ndarray.size_, 0);
    EXPECT_EQ(ndarray.buffer, nullptr);
    EXPECT_EQ(ndarray.bases_.size(), 1);
    EXPECT_EQ(ndarray.bases_[0], 0);
}

TEST(Ndarray_constructors, initializer_constructor_works){
    // Arrange
    vector<size_t > v = {1, 2};
    vector<size_t > v1 = {0};

    // Act

    // Assert
    EXPECT_NO_THROW(Ndarray<int> ndarray(v));
    EXPECT_NO_THROW(Ndarray<int> ndarray(v1));
}

TEST(Ndarray_constructors, initializer_constructor){
    // Arrange
    vector<size_t > v = {1, 2};
    vector<size_t > v1 = {0};

    // Act
    Ndarray<int> ndarray(v);
    Ndarray<int> ndarray1(v1);

    // Assert
    EXPECT_EQ(ndarray.shape_, v);
    EXPECT_EQ(ndarray.shape_.size(), 2);
    EXPECT_EQ(ndarray.size_, 2);
    EXPECT_EQ(ndarray.buffer[0], 0);
    EXPECT_EQ(ndarray.bases_.size(), 2);
    EXPECT_EQ(ndarray.bases_[0], 2);
    EXPECT_EQ(ndarray.bases_[1], 1);

    EXPECT_EQ(ndarray1.shape_, v1);
    EXPECT_EQ(ndarray1.shape_.size(), 1);
    EXPECT_EQ(ndarray1.size_, 0);
    EXPECT_EQ(ndarray1.buffer, nullptr);
    EXPECT_EQ(ndarray1.bases_.size(), 1);
    EXPECT_EQ(ndarray1.bases_[0], 0);
}

TEST(Ndarray_constructors, copy_constructor_works){
    // Arrange
    Ndarray<int> ndarray({2,3});

    // Act
    ndarray({0, 0}) = 1;
    ndarray({1, 2}) = 1;

    // Assert
    EXPECT_NO_THROW(Ndarray<int> ndarray1(ndarray));
}

TEST(Ndarray_constructors, copy_constructor){
    // Arrange
    Ndarray<int> ndarray({2,3});

    // Act
    ndarray({0, 0}) = 1;
    ndarray({1, 2}) = 2;
    Ndarray<int> ndarray1(ndarray);

    // Assert
    EXPECT_EQ(ndarray1.size_, 6);
    EXPECT_EQ(ndarray1.shape_[0], 2);
    EXPECT_EQ(ndarray1.shape_[1], 3);
    EXPECT_EQ(ndarray1.shape_.size(), 2);
    EXPECT_EQ(ndarray1.buffer[0], 1);
    EXPECT_EQ(ndarray1.buffer[5], 2);
    EXPECT_EQ(ndarray1.bases_.size(), 2);
    EXPECT_EQ(ndarray1.bases_[0], 3);
    EXPECT_EQ(ndarray1.bases_[1], 1);
}

TEST(Ndarray_constructors, initial_vector_works){
    // Arrange
    vector<int> t(100);
    t[4] = 10;

    // Act

    // Assert
    EXPECT_NO_THROW(Ndarray<int> ndarray({10,10}, t));
}

TEST(Ndarray_constructors, initial_vector){
    // Arrange
    vector<int> t(100);
    t[4] = 10;

    // Act
    Ndarray<int> ndarray({10,10}, t);

    // Assert
    EXPECT_EQ(ndarray.size_, 100);
    EXPECT_EQ(ndarray.shape_[0], 10);
    EXPECT_EQ(ndarray.shape_[1], 10);
    EXPECT_EQ(ndarray.shape_.size(), 2);
    EXPECT_EQ(ndarray.buffer[4], 10);
    EXPECT_EQ(ndarray.bases_.size(), 2);
    EXPECT_EQ(ndarray.bases_[0], 10);
    EXPECT_EQ(ndarray.bases_[1], 1);
}

TEST(Ndarray_constructors, initial_vector_wrong_size){
    // Arrange
    vector<int> t(50);

    // Act
    t[4] = 10;

    // Assert
    EXPECT_ANY_THROW(Ndarray<int> ndarray({10,10}, t));
}

TEST(Ndarray_constructors, initial_array_works){
    // Arrange
    int* t = new int[100];
    t[4] = 10;

    // Act

    // Assert
    EXPECT_NO_THROW(Ndarray<int> ndarray({10,10}, t));
}

TEST(Ndarray_constructors, initial_array_correct){
    // Arrange
    int* t = new int[100];
    t[4] = 10;

    // Act
    Ndarray<int> ndarray({10,10}, t);

    // Assert
    EXPECT_EQ(ndarray.size_, 100);
    EXPECT_EQ(ndarray.shape_[0], 10);
    EXPECT_EQ(ndarray.shape_[1], 10);
    EXPECT_EQ(ndarray.shape_.size(), 2);
    EXPECT_EQ(ndarray.buffer[4], 10);
    EXPECT_EQ(ndarray.bases_.size(), 2);
    EXPECT_EQ(ndarray.bases_[0], 10);
    EXPECT_EQ(ndarray.bases_[1], 1);
}

TEST_F(Ndarray_Methods, argmax_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(B.argmax());
}

TEST_F(Ndarray_Methods, argmax){
    // Arrange
    // Act
    B({0,1,2}) = 10;
    auto index = B.argmax();

    // Assert
    EXPECT_EQ(index, 5);
}

TEST_F(Ndarray_Methods, get_nd_index_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(B.get_nd_index(5));
}

TEST_F(Ndarray_Methods, get_nd_index_test_0){
    // Arrange
    vector<size_t > v = {0,0,0};

    // Act
    auto index = B.get_nd_index(0);

    // Assert
    EXPECT_EQ(index, v);
}

TEST_F(Ndarray_Methods, get_nd_index_test_1){
    // Arrange
    vector<size_t > v = {0,0,1};

    // Act
    auto index = B.get_nd_index(1);

    // Assert
    EXPECT_EQ(index, v);
}

TEST_F(Ndarray_Methods, get_nd_index_test_2){
    // Arrange
    vector<size_t > v = {0,0,2};

    // Act
    auto index = B.get_nd_index(2);

    // Assert
    EXPECT_EQ(index, v);
}

TEST_F(Ndarray_Methods, get_nd_index_test_5){
    // Arrange
    vector<size_t > v = {0,1,2};

    // Act
    auto index = B.get_nd_index(5);

    // Assert
    EXPECT_EQ(index, v);
}

TEST_F(Ndarray_Methods, get_nd_index_test_13){
    // Arrange
    vector<size_t > v = {1,1,1};

    // Act
    auto index = B.get_nd_index(13);

    // Assert
    EXPECT_EQ(index, v);
}

TEST_F(Ndarray_Methods, get_nd_index_wrong_index){
    // Arrange

    // Act

    // Assert
    EXPECT_ANY_THROW(B.get_nd_index(-1));
    EXPECT_ANY_THROW(B.get_nd_index(100));
}

TEST_F(Ndarray_Methods, argmax_axis_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.argmax(0));
}

TEST_F(Ndarray_Methods, argmax_axis_0){
    // Arrange
    vector<size_t > shape = {2};

    // Act
    auto index = A.argmax(0);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0}), 0);
    EXPECT_EQ(index({1}), 1);
}

TEST_F(Ndarray_Methods, argmax_axis_1){
    // Arrange
    vector<size_t > shape = {2};

    // Act
    auto index = A.argmax(1);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0}), 1);
    EXPECT_EQ(index({1}), 1);
}

TEST_F(Ndarray_Methods, argmax_axis_2){
    // Arrange
    vector<size_t > shape = {3, 3};

    // Act
    B({0,0,0}) = 10;
    B({0,1,2}) = 10;

    auto index = B.argmax(2);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0, 0}), 0);
    EXPECT_EQ(index({0, 1}), 2);
}


TEST_F(Ndarray_Methods, argmin_axis_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.argmin(0));
}

TEST_F(Ndarray_Methods, argmin_axis_0){
    // Arrange
    vector<size_t > shape = {2};

    // Act
    auto index = A.argmin(0);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0}), 1);
    EXPECT_EQ(index({1}), 0);
}

TEST_F(Ndarray_Methods, argmin_axis_1){
    // Arrange
    vector<size_t > shape = {2};

    // Act
    auto index = A.argmin(1);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0}), 0);
    EXPECT_EQ(index({1}), 0);
}

TEST_F(Ndarray_Methods, argmin_axis_2){
    // Arrange
    vector<size_t > shape = {3, 3};

    // Act
    B({0,0,0}) = 10;
    B({0,1,2}) = 10;

    auto index = B.argmin(2);

    // Assert
    EXPECT_EQ(index.shape_, shape);
    EXPECT_EQ(index({0, 0}), 1);
    EXPECT_EQ(index({0, 1}), 0);
}

TEST_F(Ndarray_Methods, max_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.max());
}

TEST_F(Ndarray_Methods, max_correct){
    // Arrange
    // Act
    auto max = A.max();

    // Assert
    EXPECT_EQ(max, 11);
}

TEST_F(Ndarray_Methods, min_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.min());
}

TEST_F(Ndarray_Methods, min_correct){
    // Arrange
    // Act
    auto min = A.min();

    // Assert
    EXPECT_EQ(min, 2);
}

TEST_F(Ndarray_Methods, max_axis_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.max(0));
}
TEST_F(Ndarray_Methods, max_axis_correct){
    // Arrange
    // Act
    auto max = A.max(0);

    // Assert
    EXPECT_EQ(max.shape_.size(), 1);
    EXPECT_EQ(max.shape_[0], 2);
    EXPECT_EQ(max({0}), 5);
    EXPECT_EQ(max({1}), 11);
}

TEST_F(Ndarray_Methods, min_axis_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.min(0));
}

TEST_F(Ndarray_Methods, min_axis_correct){
    // Arrange
    // Act
    auto max = A.min(0);

    // Assert
    EXPECT_EQ(max.shape_.size(), 1);
    EXPECT_EQ(max.shape_[0], 2);
    EXPECT_EQ(max({0}), 2);
    EXPECT_EQ(max({1}), 10);
}

TEST_F(Ndarray_Methods, reshape_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.reshape({1,4}));
}

TEST_F(Ndarray_Methods, reshape_correct_1_4){
    // Arrange
    vector<size_t > v = {1, 4};

    // Act
    A.reshape({1,4});

    // Assert
    EXPECT_EQ(A.shape(), v);
}

TEST_F(Ndarray_Methods, reshape_correct_4_1){
    // Arrange
    vector<size_t > v = {4, 1};

    // Act
    A.reshape({4,1});

    // Assert
    EXPECT_EQ(A.shape(), v);
}

TEST_F(Ndarray_Methods, reshape_correct_4_unnkown){
    // Arrange
    vector<size_t > v = {4, 1};

    // Act
    A.reshape({4,-1});

    // Assert
    EXPECT_EQ(A.shape(), v);
}

TEST_F(Ndarray_Methods, reshape_correct_unnkown_4){
    // Arrange
    vector<size_t > v = {1, 4};

    // Act
    A.reshape({-1, 4});

    // Assert
    EXPECT_EQ(A.shape(), v);
}

TEST_F(Ndarray_Methods, reshape_correct_1_unnkown_4){
    // Arrange
    vector<size_t > v = {1, 1, 4};

    // Act
    A.reshape({1, -1, 4});

    // Assert
    EXPECT_EQ(A.shape(), v);
}

TEST_F(Ndarray_Methods, reshape_wrong_shape_1_3){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(A.reshape({1,3}));
}

TEST_F(Ndarray_Methods, reshape_wrong_shape_1_unnkown_3){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(A.reshape({1,-1,3}));
}

TEST_F(Ndarray_Methods, reshape_wrong_shape_1_1){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(A.reshape({1,-1,3}));
}

TEST_F(Ndarray_Methods, fill_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(A.fill(5));
}

TEST_F(Ndarray_Methods, fill_correct){
    // Arrange
    // Act
    A.fill(5);

    // Assert
    EXPECT_EQ(A({0,0}), 5);
    EXPECT_EQ(A({0,1}), 5);
    EXPECT_EQ(A({1,0}), 5);
    EXPECT_EQ(A({1,1}), 5);
}

TEST_F(Ndarray_Methods, flatten_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(B.flatten());
}

TEST_F(Ndarray_Methods, flatten_correct){
    // Arrange
    // Act
    auto arr = B.flatten();

    // Assert
    EXPECT_EQ(arr.size_, B.size_);
    EXPECT_EQ(arr.shape_.size(), 1);
    EXPECT_EQ(arr.shape_[0], B.size_);
    for(int i = 0; i < 27; i++){
        EXPECT_EQ(arr.buffer[i], B.buffer[i]);
    }
}

TEST_F(Ndarray_Methods, indexation_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(B({0,0,0}));
}

TEST_P(Ndarray_Methods_Turple, indexation_all_right){
    // Arrange
    // Act
    // Assert
    EXPECT_EQ(B({std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam())}), 1);
}

TEST_F(Ndarray_Methods, indexation_wrong_number_param){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(B({1,1}));
    EXPECT_ANY_THROW(B({1}));
}

TEST_F(Ndarray_Methods, indexation_too_much_param){
    // Arrange
    // Act
    // Assert
    EXPECT_ANY_THROW(B({1,1,1,1}));
}

INSTANTIATE_TEST_CASE_P(
        indexation_Ndarray,
        Ndarray_Methods_Turple,
        ::testing::Values(std::make_tuple(0,0,0),
                          std::make_tuple(0,0,1),
                          std::make_tuple(0,0,2),
                          std::make_tuple(0,1,0),
                          std::make_tuple(0,1,1),
                          std::make_tuple(0,1,2),
                          std::make_tuple(0,2,0),
                          std::make_tuple(0,2,1),
                          std::make_tuple(0,2,2),
                          std::make_tuple(1,0,0),
                          std::make_tuple(1,0,1),
                          std::make_tuple(1,0,2),
                          std::make_tuple(1,1,0),
                          std::make_tuple(1,1,1),
                          std::make_tuple(1,1,2),
                          std::make_tuple(1,2,0),
                          std::make_tuple(1,2,1),
                          std::make_tuple(1,2,2),
                          std::make_tuple(2,0,0),
                          std::make_tuple(2,0,1),
                          std::make_tuple(2,0,2),
                          std::make_tuple(2,1,0),
                          std::make_tuple(2,1,1),
                          std::make_tuple(2,1,2),
                          std::make_tuple(2,2,0),
                          std::make_tuple(2,2,1),
                          std::make_tuple(2,2,2))
);