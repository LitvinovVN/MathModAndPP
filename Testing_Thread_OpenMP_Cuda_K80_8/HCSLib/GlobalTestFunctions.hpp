#pragma once


/////////////////////////////////
/// Проверка работы VectorGpu ///
bool TestVectorGpu()
{
    // Добавить разные тесты
    try
    {
        VectorGpu<double> v1{350000};        
        v1.InitByVal(0.001);
        //v1.Print();        
    
        for(int i = 1; i <= 5; i++)
        {
            for(int j = 1; j <= 5; j++)
            {
                auto res = ArrayHelper::SumCuda(v1.Get_dev_data_pointer(), v1.Size(),i,j);
                std::cout << i << ", " << j << ": ";
                //res.Print();
                std::cout << res << std::endl;
            }
        }

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
}
///////////////////////////////////////////////////////////
/// Тестирование функции суммирования элементов массива ///
bool TestSum()
{
    TestParams testParams;
    testParams.IterNum = 1;

    // 1. Подготовка данных
    unsigned Nthreads = 10;
    size_t size = 1000000000;
    double elVal = 0.001;
    VectorRam<double> v(size);
    v.InitByVal(elVal);
    //v.Print();

    VectorGpu<double>* vGpu_p = nullptr;
    try
    {
        vGpu_p = new VectorGpu<double>(size);
        vGpu_p->InitByVal(elVal);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    // 2. Запуск тестов и получение массива результатов
    // 2.1 Последовательный алгоритм
    auto testResults_seq = TestHelper::LaunchSum(v, testParams);
    std::cout << "Seq: testResults_seq size = " << testResults_seq.size() << std::endl;
    for(auto& res : testResults_seq)
        res.Print();
    // 2.2 Параллельный алгоритм std::thread
    auto testResults_par = TestHelper::LaunchSum(v, Nthreads, testParams);
    std::cout << "Parallel: testResults size = " << testResults_par.size() << std::endl;
    for(auto& res : testResults_par)
        res.Print();
    // 2.3 Параллельный алгоритм OpenMP
    auto testResults_par_OpenMP = TestHelper::LaunchSumOpenMP(v, Nthreads, testParams);
    std::cout << "Parallel OpenMP: testResults size = " << testResults_par_OpenMP.size() << std::endl;
    for(auto& res : testResults_par_OpenMP)
        res.Print();

    // 2.4 Параллельный алгоритм Cuda
    int numBlocks = 10;
    auto testResults_par_Cuda = TestHelper::LaunchSumCuda(*vGpu_p, numBlocks, Nthreads, testParams);
    std::cout << "Parallel CUDA: testResults size = " << testResults_par_Cuda.size() << std::endl;
    for(auto& res : testResults_par_Cuda)
        res.Print();

    // 2.5 Параллельный алгоритм Cuda на 1 GPU с двумя видеочипами
    //int numBlocks = 37;
    /*auto testResults_par2_Cuda = TestHelper::LaunchSumCudaMultiGpu(testParamsGpu);
    std::cout   << "Parallel CUDA LaunchSumCudaMultiGpu: testResults size = "
                << testResults_par2_Cuda.size() << std::endl;
    for(auto& res : testResults_par2_Cuda)
        res.Print();*/

    // Освобождаем видеопамять
    vGpu_p->Clear_dev_data();

    // 3. Статистическая обработка результатов
    CalculationStatistics stat_seq{testResults_seq};
    std::cout << "CalculationStatistics seq: " << std::endl;
    stat_seq.Print();

    CalculationStatistics stat_par{testResults_par};
    std::cout << "CalculationStatistics parallel std::thread: " << std::endl;
    stat_par.Print();

    CalculationStatistics stat_par_OpenMP;
    try
    {
        stat_par_OpenMP = CalculationStatistics{testResults_par_OpenMP};
        std::cout << "CalculationStatistics parallel OpenMP: " << std::endl;
        stat_par_OpenMP.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';        
    }

    CalculationStatistics stat_par_Cuda;
    try
    {
        stat_par_Cuda = CalculationStatistics{testResults_par_Cuda};
        std::cout << "CalculationStatistics parallel Cuda: " << std::endl;
        stat_par_Cuda.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    /*CalculationStatistics stat_par2_Cuda;
    try
    {
        stat_par2_Cuda = CalculationStatistics{testResults_par2_Cuda};
        std::cout << "CalculationStatistics parallel Cuda LaunchSumCudaDevNum1GpuNum2: " << std::endl;
        stat_par2_Cuda.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }*/
    

    // 4. Вычисляем ускорение и эффективность
    std::cout << "--- std::thread ---" << std::endl;
    ParallelCalcIndicators parallelCalcIndicators(stat_seq, stat_par, Nthreads);
    parallelCalcIndicators.Print();

    try
    {
        std::cout << "--- OpenMP ---" << std::endl;
        ParallelCalcIndicators parallelCalcIndicators_OpenMP(stat_seq, stat_par_OpenMP, Nthreads);
        parallelCalcIndicators_OpenMP.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    try
    {
        std::cout << "--- CUDA ---" << std::endl;
        ParallelCalcIndicators parallelCalcIndicators_Cuda(stat_seq, stat_par_Cuda, numBlocks*Nthreads);
        parallelCalcIndicators_Cuda.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    /*try
    {
        std::cout << "--- CUDA, 1 dev, 2 videochips ---" << std::endl;
        ParallelCalcIndicators parallelCalcIndicators_Cuda2(stat_seq, stat_par2_Cuda, numBlocks*Nthreads);
        parallelCalcIndicators_Cuda2.Print();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }*/

    return true;
}
