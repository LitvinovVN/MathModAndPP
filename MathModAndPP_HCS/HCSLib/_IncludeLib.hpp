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

#ifdef OPENBLAS
#include <cblas.h>  // OpenBLAS
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

///// Вспомогательные типы
#include "CommonHelpers/_IncludeCommonHelpers.hpp"

///// Дифференциальные уравнения
#include "DifferentialEquations/_IncludeDifferentialEquations.hpp"

///// Геометрия расчетной области
#include "Geometry/_IncludeGeometry.hpp"


///// Параметры проведения тестов производительности (начало) /////
#include "PerformanceTests/PerfTestParamsData.hpp"
#include "PerformanceTests/PerfTestParamsCpu.hpp"
#include "PerformanceTests/PerfTestParamsGpu.hpp"
#include "PerformanceTests/PerfTestParams.hpp"
#include "PerformanceTests/CalculationStatistics.hpp"
#include "PerformanceTests/ParallelCalcIndicators.hpp"
#include "PerformanceTests/PerfTestResults.hpp"
///// Параметры проведения тестов производительности (конец) /////

///// Модуль Math (начало) /////
#include "Math/MathObject.hpp"
#include "Math/Expression.hpp"
#include "Math/Constant.hpp"
#include "Math/Variable.hpp"
#include "Math/Negate.hpp"
#include "Math/BinaryExpression.hpp"
#include "Math/FuncExpression.hpp"
#include "Math/GridContext.hpp"
#include "Math/GridEvaluableObject.hpp"
#include "Math/GridOperator.hpp"
#include "Math/GridOperatorEvaluator.hpp"
#include "Math/MathHelper.hpp"
#include "Math/MathHelper_ConsoleUI.hpp"
///// Модуль Math (конец) /////

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
#include "Arrays/ArrayHelperFuncResult.hpp"
#include "Arrays/ArrayPerfTestHelper.hpp"
#include "Arrays/ArrayPerfTestHelper_ConsoleUI.hpp"
#include "Arrays/ArrayBlockIndexes.hpp"
#include "Arrays/ArraysIndexMap.hpp"
#include "Arrays/DevMemArrPointer.hpp"
#include "Arrays/DevMemArrPointers.hpp"
////////// Массивы (конец) ////////////

////////// Модуль Scalars (Скалярные значения) ////////////
#include "Scalars/_IncludeScalars.hpp"

////////// Модуль Vectors (Векторы) ////////////
#include "Vectors/_IncludeVectors.hpp"

////////// Матрицы (начало) ////////////
#include "Matrices/MatrixDataLocation.hpp"
#include "Matrices/IMatrix.hpp"
#include "Matrices/MatrixRam.hpp"
#include "Matrices/MatrixRamZero.hpp"
#include "Matrices/MatrixRamE.hpp"
#include "Matrices/MatrixBlockRamGpus.hpp"
#include "Matrices/MatrixMapElement.hpp"
#include "Matrices/MatrixMap.hpp"
#include "Matrices/MatricesHelper.hpp"
#include "Matrices/MatricesHelper_ConsoleUI.hpp"
////////// Матрицы (конец) ////////////

#include "TestParams.hpp"
#include "TestHelper.hpp"


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
#include "Algorithms/AlgorithmDataLocation.hpp"
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

///// Решения модельных задач
#include "ModelProblems/_IncludeModelProblems.hpp"

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