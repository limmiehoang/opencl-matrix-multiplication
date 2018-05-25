#include "matmul.hpp"
#include "matrix_lib.hpp"
#include "err_code.h"
#include "device_picker.hpp"

int main(int argc, char *argv[])
{

    int N;      // A[N][N], B[N][N], C[N][N]
    int size;   // Number of elements in each matrix


    double start_time;      // Starting time
    double run_time;        // Timing data
    util::Timer timer;      // Timer


    N = ORDER;

    size = N * N;

    std::vector<float> h_A(size); // Host memory for Matrix A
    std::vector<float> h_B(size); // Host memory for Matrix B
    std::vector<float> h_C(size); // Host memory for Matrix C

    cl::Buffer d_a, d_b, d_c;   // Matrices in device memory

//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {

        cl_uint deviceIndex = 1;
        parseArguments(argc, argv, &deviceIndex);



        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

//--------------------------------------------------------------------------------
// Run sequential matrix multiplication
//--------------------------------------------------------------------------------

        initmat(N, h_A, h_B, h_C);

        //printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
		printf("\n===== Sequential, matrix multiplication [1024][1024]*[1024][1024] on host CPU ======\n");
        for(int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);
            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // seq_mat_mul_sdot(N, h_A, h_B, h_C);

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            results(N, h_C, run_time);
        }

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

        //  Reset A, B and C matrices (just to play it safe)
        initmat(N, h_A, h_B, h_C);

        d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);

        d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Common Optimization
//--------------------------------------------------------------------------------

        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("optimize_common.cl"), true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program, "mmul");

        //printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",N);
		printf("\n===== OpenCL, matrix multiplication [1024][1024]*[1024][1024], optimized per work item ======\n");
        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(N, N);
            naive_mmul(cl::EnqueueArgs(queue, global),
                    N, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            results(N, h_C, run_time);

        } // end for loop

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Optimize: One work item per row
//--------------------------------------------------------------------------------

		program = cl::Program(context, util::loadProgram("optimize_per_row.cl"), true);

		// Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>row_mmul(program, "mmul");

        //printf("\n===== OpenCL, matrix mult, C(i,j) optimize_per_row, order %d ======\n",N);
		printf("\n===== OpenCL, matrix multiplication [1024][1024]*[1024][1024], optimized per row ======\n");
        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(N);
            row_mmul(cl::EnqueueArgs(queue, global),
                    N, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            results(N, h_C, run_time);

        } // end for loop


//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Optimize: Row of A in private memory
//--------------------------------------------------------------------------------

		program = cl::Program(context, util::loadProgram("optimize_private.cl"), true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> private_mmul(program, "mmul");

        //printf("\n===== OpenCL, matrix mult, C(i,j) optimize_private, order %d ======\n",N);
		printf("\n===== OpenCL, matrix multiplication [1024][1024]*[1024][1024], optimized private ======\n");
        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(N);
			cl::NDRange local(ORDER/16);
            private_mmul(cl::EnqueueArgs(queue, global,local),
                    N, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            results(N, h_C, run_time);

        } // end for loop

    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
