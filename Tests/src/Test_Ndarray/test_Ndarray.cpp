#include "gtest/gtest.h"
#define TEST_Ndarray
#include "Ndarray.h"

using namespace NN;

class Ndarray_Methods : public ::testing::Test {
public:
    Ndarray_Methods(): B({3,3,3}) {}

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
    }
    void TearDown() { /* called after every test */ }
public:
    Ndarray<double> B;
};
class Ndarray_Methods_P : public ::testing::TestWithParam<int> {
public:
    Ndarray_Methods_P(): B({3,3,3}) {}

    ~Ndarray_Methods_P() { /* free protected members here */ }

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

    EXPECT_EQ(ndarray1.shape_, v1);
    EXPECT_EQ(ndarray1.shape_.size(), 1);
    EXPECT_EQ(ndarray1.size_, 0);
    EXPECT_EQ(ndarray1.buffer, nullptr);
}

//TEST(Ndarray_constructors, copy_constructor_works){
//    // Arrange
//    Ndarray<int> ndarray({2,3});
//
//    // Act
//
//    // Assert
//    EXPECT_NO_THROW(Ndarray<int> ndarray(v));
//    EXPECT_NO_THROW(Ndarray<int> ndarray(v1));
//}

TEST_F(Ndarray_Methods, indexation_works){
    // Arrange
    // Act
    // Assert
    EXPECT_NO_THROW(B(0,0,0));
}

TEST_P(Ndarray_Methods_P, indexation_all_right){
    // Arrange
    // Act
    // Assert
    EXPECT_EQ(B(GetParam(),GetParam(),GetParam()), 1);
}

INSTANTIATE_TEST_CASE_P(
        indexation_Ndarray,
        Ndarray_Methods_P,
        ::testing::Range(1,3)
        );

TEST_P(Ndarray_Methods_P, indexation_wrong_param){
    // Arrange
    // Act
    // Assert
    EXPECT_EQ(B(GetParam(),GetParam()), 1);
}