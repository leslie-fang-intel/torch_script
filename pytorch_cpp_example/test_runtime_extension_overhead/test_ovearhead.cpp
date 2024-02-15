
#include "TaskExecutor.h"
#include "Task.h"
#include "TaskModule.h"

// for test of performance overhead
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/util/Exception.h>

unsigned cycles_low1, cycles_high1, cycles_low2, cycles_high2;
unsigned cycles_low3, cycles_high3, cycles_low4, cycles_high4;
uint64_t timestamp1, timestamp2, timestamp3, timestamp4;

std::chrono::time_point<std::chrono::high_resolution_clock> timestamp1_nano = std::chrono::high_resolution_clock::now();
std::chrono::time_point<std::chrono::high_resolution_clock> timestamp2_nano = std::chrono::high_resolution_clock::now();

std::condition_variable cv;
std::mutex cv_m; // This mutex is used for three purposes:
                 // 1) to synchronize accesses to i
                 // 2) to synchronize accesses to std::cerr
                 // 3) for the condition variable cv
bool ready = false;

namespace torch_ipex {
namespace runtime {

int taskfunction(int i) {
	//std::cout << "taskfunction thread num : " << std::this_thread::get_id() << ", arg : " << i << std::endl;
	return i;
}

// for test of performance overhead
at::Tensor taskfunction3(at::Tensor input) {
    at::Tensor output;
    //for (size_t i = 0; i < 50000; i++) {
    for (size_t i = 0; i < 1; i++) {
        output = at::softmax(input, -1);
    }
    return input;
}

void test_threadpool_overhead() {
    std::cout<<"ipex inside test_threadpool_overhead"<<std::endl;
#if defined(ENABLE_RUNTIME_EXT)
    std::vector<int32_t> cpu_core_list({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    std::shared_ptr<TaskExecutor> task_executor = std::make_shared<TaskExecutor>(cpu_core_list);
    at::Tensor input_tensor = at::rand({100, 8276});
    //Task<at::Tensor (*)(at::Tensor), at::Tensor> b(taskfunction3, task_executor);

    Task<int (*)(int), int> b(taskfunction, task_executor);

    // Warm up time measurement
    asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");

    asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");

    asm volatile ("CPUID\n\t"
    "RDTSC\n\t"::: "%rax", "%rbx", "%rcx", "%rdx");

    std::this_thread::sleep_for(std::chrono::seconds(2)); // to test time, we need to make sure thread pool finish init

    // Measure time1
    asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");

    timestamp1_nano = std::chrono::high_resolution_clock::now();

    //auto resf = b(std::move(input_tensor));
    auto resf = b(1);
    //std::cout<<"waiting to get result"<<std::endl;
    auto res = resf.get();

    // Measure time4
    asm volatile ( "CPUID\n\t"
            "RDTSC\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t": "=r" (cycles_high4), "=r" (cycles_low4)::"%rax", "%rbx", "%rcx", "%rdx");

    timestamp1 = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );
    timestamp2 = ( ((uint64_t)cycles_high2 << 32) | cycles_low2 );
    timestamp3 = ( ((uint64_t)cycles_high3 << 32) | cycles_low3 );
    timestamp4 = ( ((uint64_t)cycles_high4 << 32) | cycles_low4 );

    std::cout<<"submit time clock(timestamp2-timestamp1): "<<timestamp2-timestamp1<<std::endl;
    std::cout<<"execution time clock(timestamp3-timestamp2): "<<timestamp3-timestamp2<<std::endl;
    std::cout<<"join time clock(timestamp4-timestamp3): "<<timestamp4-timestamp3<<std::endl;

    float submit_time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            timestamp2_nano - timestamp1_nano).count() / 1000.0;
    std::cout<<"submit_time_us is: "<<submit_time_us<<" us"<<std::endl;
#else
    std::cout<<"runtime threadpool build flag is not enabled"<<std::endl;
#endif
    return;
}

void waits()
{
    for (int i =0; i<10; i++) {
        std::unique_lock<std::mutex> lk(cv_m);
        std::cerr << "Waiting... \n";
        cv.wait(lk, []{return ready == true;});
        asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::"%rax", "%rbx", "%rcx", "%rdx");
        ready = false;
    }
    std::cerr << "...finished waiting. i == 1\n";
}
 
void signals()
{ 
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i =0; i<10; i++) {
        {
            std::lock_guard<std::mutex> lk(cv_m);
            //i = 1;
            ready = true;
            std::cerr << "Notifying again...\n";
        }
        asm volatile ( "CPUID\n\t"
        "RDTSC\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");
        cv.notify_one();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void test_cv_overhead() {
    std::cout<<"ipex inside test_cv_overhead"<<std::endl;
#if defined(ENABLE_RUNTIME_EXT)
    asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");

    asm volatile ( "CPUID\n\t"
                "RDTSC\n\t"
                "CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");

    asm volatile ("CPUID\n\t"
    "RDTSC\n\t"::: "%rax", "%rbx", "%rcx", "%rdx");

    std::thread t1(waits), t4(signals);
    t1.join(); 
    t4.join();

    timestamp1 = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );
    timestamp2 = ( ((uint64_t)cycles_high2 << 32) | cycles_low2 );
    std::cout<<timestamp1<<std::endl;
    std::cout<<timestamp2<<std::endl;
    std::cout<<"timestamp2 - timestamp1(submit time clock): "<<timestamp2-timestamp1<<std::endl;
#else
    std::cout<<"runtime threadpool build flag is not enabled for test_cv_overhead"<<std::endl;
#endif
    return;
}

void test_multitask_submition() {
    std::cout<<"ipex inside test_multitask_submition"<<std::endl;
    std::vector<int32_t> cpu_core_list1({0, 1, 2, 3});
    std::vector<int32_t> cpu_core_list2({4, 5, 6, 7});
    std::vector<int32_t> cpu_core_list3({8, 9, 10, 11});
    std::vector<int32_t> cpu_core_list4({12, 13, 14, 15});
    std::vector<int32_t> cpu_core_list5({16, 17, 18, 19});
    std::vector<int32_t> cpu_core_list6({20, 21, 22, 23});
    std::vector<int32_t> cpu_core_list7({24, 25, 26, 27});

    std::shared_ptr<TaskExecutor> task_executor1 = std::make_shared<TaskExecutor>(cpu_core_list1);
    std::shared_ptr<TaskExecutor> task_executor2 = std::make_shared<TaskExecutor>(cpu_core_list2);
    std::shared_ptr<TaskExecutor> task_executor3 = std::make_shared<TaskExecutor>(cpu_core_list3);
    std::shared_ptr<TaskExecutor> task_executor4 = std::make_shared<TaskExecutor>(cpu_core_list4);
    std::shared_ptr<TaskExecutor> task_executor5 = std::make_shared<TaskExecutor>(cpu_core_list5);
    std::shared_ptr<TaskExecutor> task_executor6 = std::make_shared<TaskExecutor>(cpu_core_list6);
    std::shared_ptr<TaskExecutor> task_executor7 = std::make_shared<TaskExecutor>(cpu_core_list7);

    Task<at::Tensor (*)(at::Tensor), at::Tensor> task1(taskfunction3, task_executor1);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task2(taskfunction3, task_executor2);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task3(taskfunction3, task_executor3);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task4(taskfunction3, task_executor4);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task5(taskfunction3, task_executor5);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task6(taskfunction3, task_executor6);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task7(taskfunction3, task_executor7);

    for(int i=0; i<100; i++) {
        at::Tensor input_tensor = at::rand({100, 8276});
        at::Tensor input_tensor2 = at::rand({100, 8276});
        at::Tensor input_tensor3 = at::rand({100, 8276});
        at::Tensor input_tensor4 = at::rand({100, 8276});
        at::Tensor input_tensor5 = at::rand({100, 8276});
        at::Tensor input_tensor6 = at::rand({100, 8276});
        at::Tensor input_tensor7 = at::rand({100, 8276});

        auto resf1 = task1(std::move(input_tensor));
        auto resf2 = task2(std::move(input_tensor2));
        auto resf3 = task3(std::move(input_tensor3));
        auto resf4 = task4(std::move(input_tensor4));
        auto resf5 = task5(std::move(input_tensor5));
        auto resf6 = task6(std::move(input_tensor6));
        auto resf7 = task7(std::move(input_tensor7));
        //std::cout<<"waiting to get result"<<std::endl;
        //std::this_thread::sleep_for(std::chrono::seconds(2));
        auto res1 = resf1.get();
        auto res2 = resf2.get();
        auto res3 = resf3.get();
        auto res4 = resf4.get();
        auto res5 = resf5.get();
        auto res6 = resf6.get();
        auto res7 = resf7.get();
    }
    return;
}

TORCH_API void test_script_module_multitask(const torch::jit::Module& module, py::args args, py::kwargs kwargs) {
    std::cout<<"ipex inside test_script_module_multitask"<<std::endl;
    std::vector<int32_t> cpu_core_list1({0, 1, 2, 3});
    std::vector<int32_t> cpu_core_list2({4, 5, 6, 7});
    std::vector<int32_t> cpu_core_list3({8, 9, 10, 11});
    std::vector<int32_t> cpu_core_list4({12, 13, 14, 15});
    std::vector<int32_t> cpu_core_list5({16, 17, 18, 19});
    std::vector<int32_t> cpu_core_list6({20, 21, 22, 23});
    std::vector<int32_t> cpu_core_list7({24, 25, 26, 27});

    torch_ipex::runtime::TaskModule task1(module, std::move(cpu_core_list1), true);
    torch_ipex::runtime::TaskModule task2(module.deepcopy(), std::move(cpu_core_list2), true);
    torch_ipex::runtime::TaskModule task3(module.deepcopy(), std::move(cpu_core_list3), true);
    torch_ipex::runtime::TaskModule task4(module.deepcopy(), std::move(cpu_core_list4), true);
    torch_ipex::runtime::TaskModule task5(module.deepcopy(), std::move(cpu_core_list5), true);
    torch_ipex::runtime::TaskModule task6(module.deepcopy(), std::move(cpu_core_list6), true);
    torch_ipex::runtime::TaskModule task7(module.deepcopy(), std::move(cpu_core_list7), true);

    at::Tensor input_tensor = at::rand({64, 64, 3, 3});

    for(int i=0; i<10; i++) {
        py::args args1 = args;
        py::kwargs kwargs1 = kwargs;

        py::args args2 = args;
        py::kwargs kwargs2 = kwargs;

        py::args args3 = args;
        py::kwargs kwargs3 = kwargs;

        py::args args4 = args;
        py::kwargs kwargs4 = kwargs;

        py::args args5 = args;
        py::kwargs kwargs5 = kwargs;

        py::args args6 = args;
        py::kwargs kwargs6 = kwargs;

        py::args args7 = args;
        py::kwargs kwargs7 = kwargs;

        task1.submit(std::move(args1), std::move(kwargs1));
        task2.submit(std::move(args2), std::move(kwargs2));
        task3.submit(std::move(args3), std::move(kwargs3));
        task4.submit(std::move(args4), std::move(kwargs4));
        task5.submit(std::move(args5), std::move(kwargs5));
        task6.submit(std::move(args6), std::move(kwargs6));
        task7.submit(std::move(args7), std::move(kwargs7));

        task1.wait();
        task2.wait();
        task3.wait();
        task4.wait();
        task5.wait();
        task6.wait();
        task7.wait();
    }
    return;
}

TORCH_API void test_task_multi_submition() {
    std::cout<<"ipex inside test_task_multi_submition"<<std::endl;
    std::vector<int32_t> cpu_core_list1({0, 1, 2, 3});
    std::vector<int32_t> cpu_core_list2({4, 5, 6, 7});

    std::shared_ptr<TaskExecutor> task_executor1 = std::make_shared<TaskExecutor>(cpu_core_list1);
    std::shared_ptr<TaskExecutor> task_executor2 = std::make_shared<TaskExecutor>(cpu_core_list2);

    Task<at::Tensor (*)(at::Tensor), at::Tensor> task1(taskfunction3, task_executor1);
    Task<at::Tensor (*)(at::Tensor), at::Tensor> task2(taskfunction3, task_executor2);

    at::Tensor input_tensor = at::rand({10, 64});
    at::Tensor input_tensor2 = at::rand({10, 64});
    at::Tensor input_tensor3 = at::rand({10, 64});
    at::Tensor input_tensor4 = at::rand({10, 64});
    at::Tensor input_tensor5 = at::rand({10, 64});
    at::Tensor input_tensor6 = at::rand({10, 64});

    auto task1_resf1 = task1(std::move(input_tensor));
    auto task1_resf2 = task1(std::move(input_tensor2));
    auto task1_resf3 = task1(std::move(input_tensor3));

    auto task2_resf4 = task2(std::move(input_tensor4));
    auto task2_resf5 = task2(std::move(input_tensor5));
    auto task2_resf6 = task2(std::move(input_tensor6));

    auto task1_res1 = task1_resf1.get();
    auto task1_res2 = task1_resf2.get();
    auto task1_res3 = task1_resf3.get();

    auto task2_res4 = task2_resf4.get();
    auto task2_res5 = task2_resf5.get();
    auto task2_res6 = task2_resf6.get();

    std::cout<<task1_res1<<std::endl;
    std::cout<<task1_res2<<std::endl;

    return;
}

}}
