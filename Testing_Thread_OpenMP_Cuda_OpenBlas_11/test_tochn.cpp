// Тест точности вычисления скалярного произведения векторов
// nvcc test_tochn.cpp -o test_tochn -x cu
// ./test_tochn
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

int main()
{
    std::cout << "Test tochnosti\n";
    float f8 = 3210.12345678;
    std::cout << "f8 = 3210.12345678 = " << std::setw(10) << f8 << std::endl;

    double d8 = 3210.12345678;
    std::cout << "d8 = 3210.12345678 = " << std::setw(10) << d8 << std::endl;
    std::cout << d8 - 0.00000321012345678 << std::endl;

    
    double sum_res{0};
    std::vector<float> vec1{ -1.2345E8, 1E-7 , 1E6, -1E-7, -1E6 };
    //std::sort(std::begin(vec1), std::end(vec1));
    double sum_vec1{0};
    for(auto i{0ull}; i < vec1.size(); i++)
    {
        sum_vec1 += vec1[i];
    }
    sum_res+=sum_vec1;
    //printf("sum_res = %lf\n", sum_res);
    double local_sum{0};
    auto N = 1E8;
    printf("sum_res = %lf\n", sum_res);
    for(auto i{1ull}; i <= N; i++)
    {
        double a{10.0/i};
        double b{-1.0};
        //sum_res += a; sum_res += b;
        local_sum += a; local_sum += b;
    }
    sum_res +=1;
    printf("sum_res = %lf\n", sum_res);
    for(auto i{1ull}; i <= N; i++)
    {
        double a{1.0};
        double b{-10.0/i};
        //sum_res += a; sum_res += b;
        local_sum += a; local_sum += b;
    }
    printf("local_sum = %lf\n", local_sum);
    sum_res += local_sum;
    sum_res+=1.2345E8;
    
    printf("sum_res = %lf\n", sum_res);

}