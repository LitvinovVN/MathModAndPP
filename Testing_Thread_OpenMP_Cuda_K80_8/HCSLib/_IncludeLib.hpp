#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <map>

using namespace std::chrono;

#ifdef _OPENMP
#include <omp.h>
#endif

///////////// CUDA (начало) /////////////
#ifdef __NVCC__
#include "Cuda/kernels.cu"
#include <cublas_v2.h>
#endif
#include "Cuda/CudaDeviceProperties.hpp"
#include "Cuda/CudaHelper.hpp"
#include "Cuda/CublasHelper.hpp"
///////////// CUDA (конец) /////////////

///// Вспомогательные типы (начало) /////
#include "CommonHelpers/LibSupport.hpp"
#include "CommonHelpers/FuncResult.hpp"
#include "CommonHelpers/PrintParams.hpp"
#include "CommonHelpers/ConsoleHelper.hpp"
#include "CommonHelpers/FileSystemHelper.hpp"
///// Вспомогательные типы (конец) /////


////////// Функции (начало) ////////////
#include "Functions/FunctionDataType.hpp"
#include "Functions/FunctionDataTypes.hpp"
#include "Functions/FunctionArgument.hpp"
#include "Functions/FunctionArguments.hpp"
#include "Functions/Function.hpp"
////////// Функции (конец) ////////////

////////// Массивы (начало) ////////////
#include "Arrays/ArrayGpuProcessingParams.hpp"
#include "Arrays/ArrayHelper.hpp"
#include "Arrays/ArrayHelper_ConsoleUI.hpp"
////////// Массивы (конец) ////////////

////////// Векторы (начало) ////////////
#include "Vectors/Vector.hpp"
#include "Vectors/VectorRam.hpp"
#include "Vectors/VectorGpu.hpp"
#include "Vectors/VectorRamHelper.hpp"
#include "Vectors/VectorGpuHelper.hpp"
////////// Векторы (конец) ////////////

////////// Матрицы (начало) ////////////
#include "Matrices/IMatrix.hpp"
#include "Matrices/MatrixRam.hpp"
#include "Matrices/MatrixRamZero.hpp"
#include "Matrices/MatrixRamE.hpp"
////////// Матрицы (конец) ////////////

#include "TestParams.hpp"
#include "TestHelper.hpp"
#include "CalculationStatistics.hpp"
#include "ParallelCalcIndicators.hpp"

//////// Вычислительная система (начало) ///////
#include "ComputingSystem/RamParams.hpp"
#include "ComputingSystem/CpuParams.hpp"
#include "ComputingSystem/GpuParams.hpp"
#include "ComputingSystem/ComputingSystemNode.hpp"
#include "ComputingSystem/ComputingSystem.hpp"
#include "ComputingSystem/ComputingSystemRepository.hpp"
//////// Вычислительная система (конец) ////////

//////// Задачи (начало) ///////
#include "Tasks/TaskGroup.hpp"
#include "Tasks/Task.hpp"
#include "Tasks/TaskDimensions.hpp"
//////// Задачи (конец) ///////

/////////////// Алгоритмы (начало) ///////////////////
#include "Algorithms/DataLocation.hpp"
#include "Algorithms/AlgorithmType.hpp"
#include "Algorithms/Algorithm.hpp"
#include "Algorithms/AlgorithmRepository.hpp"
#include "Algorithms/AlgorithmMetrics.hpp"
#include "Algorithms/AlgorithmImplementation.hpp"
#include "Algorithms/AlgorithmImplementationRepository.hpp"
#include "Algorithms/AlgorithmImplementationExecParams.hpp"
#include "Algorithms/AlgorithmImplementationExecutor.hpp"
#include "Algorithms/AlgorithmImplementationExecutorHelper.hpp"
/////////////// Алгоритмы (конец) ///////////////////

//// Результаты тестовых запусков алгоритмов (начало) /////
#include "AlgTestingResults/AlgTestingResult.hpp"
#include "AlgTestingResults/AlgTestingResultRepository.hpp"
///// Результаты тестовых запусков алгоритмов (конец) /////

////////// Глобальные функции (начало) ///////////
#include "GlobalTestFunctions.hpp"
////////// Глобальные функции (конец) ////////////

///////////// Приложение (начало) ////////////////
// Конфигурация приложения
#include "AppConfig.hpp"
// Меню (начало)
#include "Menu/MenuCommand.hpp"
#include "Menu/MenuCommandItem.hpp"
#include "Menu/MenuFunctions.hpp"
#include "Menu/MainMenu.hpp"
// Меню (конец)
// Класс "Приложение"
#include "Application.hpp"
///////////// Приложение (конец) ////////////////