#include <iostream>
#include <immintrin.h>
#include <stdio.h>

#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono> 
#include <sys/wait.h> 
//#define _GNU_SOURCE
#include <sched.h>

#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>


#define COUNT 20000

cpu_set_t  mask;



inline void assignToThisCore(int core_id)
{
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

#define SHM_SIZE 1

int main(int argc, char *argv[])
{



// ---------------------------

    std::cout << "Master parent:" << std::endl;    

    int shmid;
    char *shmaddr;
    key_t key = 1234;
    pid_t pid1, pid2;

    // create the shared memory segment
    if ((shmid = shmget(key, SHM_SIZE, IPC_CREAT | 0666)) < 0) {
        perror("shmget");
        exit(1);
    }


    shmaddr = static_cast<char*>(shmat(shmid, NULL, 0));


    // named pipe
    int fd;
    char * myfifo = (char*)"mypipe";
    mkfifo(myfifo, 0666);



    // fork the first process
    if ((pid1 = fork()) == 0) 
    {
        assignToThisCore(0);
        std::cout << "master child 1 waiting for a:" << std::endl;  
        fflush(stdout); 

        

        
        auto start = std::chrono::high_resolution_clock::now();
        

        /* Initialize the two argument vectors */
        __m256 vec_in_1 = _mm256_set_ps(100.1, 200.2, 300.3, 400.4, 500.5, 600.6, 700.7, 800.8);
        __m256 vec_in_2 = _mm256_set_ps(1.01, 2.01, 4.01, 5.01, 5.01, 20.01, 10.01, 8.01);
        __m256 vec_in_3 = _mm256_set_ps(1.0, 0.5, 0.25, 0.2, 0.2, 0.05, 0.1, 0.125);

        /* Compute the difference between the two vectors */
        __m256 vec_out_1 = vec_in_1;//_mm256_div_ps(vec_in_1, vec_in_2);

        __m256 vec_out_2; //= _mm256_setzero_ps();
        //__m256 vec_out_1; //= _mm256_setzero_ps();

        long long microseconds[COUNT];

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        microseconds[0] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        //printf("elapsed time1: %llu us \n", microseconds[0]);

        // wait until the second process writes 'a'
        while (*shmaddr != 'a') {}

        std::cout << "master got a" << std::endl; 
        fflush(stdout); 

        int cnt = 100000;
        char arr2[20];
        // Open FIFO for Read only
        fd = open(myfifo, O_RDONLY);
        // Read from FIFO
        while(cnt>0)
        {
            read(fd, arr2, sizeof(arr2));
            if(arr2[0]=='x')
            {
                break;
            }
            cnt--;
        }
        // Print the read message
        if (cnt>0)
        {
            printf("User2: %s\n", arr2);
        }
        
        close(fd);


        char *inst = argv[3];

        cnt = 0;

        while(cnt<COUNT)
        {

            vec_out_1 = vec_in_1;
            start = std::chrono::high_resolution_clock::now();
            for(int i=0; i<10000; i++)
            {

                switch(*inst)
                {
                    case '1':
                        vec_out_2 = _mm256_mul_ps(vec_out_1,vec_in_2);
                        vec_out_1 = _mm256_mul_ps(vec_out_2,vec_in_3);
                        break;
                    case '2':
                        vec_out_1 = _mm256_div_ps(vec_out_1,vec_in_2);
                        //vec_out_1 = _mm256_div_ps(vec_out_2,vec_in_3);
                        break;
                    case '3':
                        vec_out_2 = _mm256_mul_ps(vec_out_1,vec_in_2);
                        vec_out_1 = _mm256_div_ps(vec_out_2,vec_in_2);
                        break;
                    case '4':
                        vec_out_2 = _mm256_add_ps(vec_out_1,vec_in_2);
                        vec_out_1 = _mm256_sub_ps(vec_out_2,vec_in_2);
                        break;
                    case '5':
                        // if the CPU does not support fma, replace this kernel with somthing else.
                        vec_out_2 = _mm256_fmadd_ps(vec_out_1,vec_in_3,vec_in_2);
                        vec_out_1 = _mm256_fmadd_ps(vec_out_2,vec_in_3,vec_in_2);
                        break;
                    case '6':
                        vec_out_1 = _mm256_permute_ps(vec_out_1,0b01110100);
                        vec_out_1 = _mm256_permute_ps(vec_out_1,0b10110101);
                        break;
                        

                    default:
                        printf("Oops");
                }



            }
            elapsed = std::chrono::high_resolution_clock::now() - start;
            microseconds[cnt] = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

            cnt++;
        }

        char *filename = argv[2];
        FILE *fptr;
        fptr = fopen(filename,"w");
        if(fptr == NULL)
        {
            printf("Error!");   
            exit(1);             
        }

        for(int i=0;i<cnt;i++)
        {
            //printf("elapsed: %llu us \n", microseconds[i]);
            //fflush(stdout); 
            fprintf(fptr,"%llu \n",microseconds[i]);
        }

        fclose(fptr);

        float* f = (float*)&vec_out_1;
        printf("%f %f %f %f %f %f %f %f\n",
            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
        fflush(stdout); 

        // exit the first process
        exit(0);



    } else if (pid1 < 0) {
        perror("fork");
        exit(1);
    }


    // fork the second process
    if ((pid2 = fork()) == 0) 
    {
        assignToThisCore(2);
        std::cout << "master child 2 runs slave" << std::endl;
        std::cout << "master child 2 write a" << std::endl;

        fflush(stdout); 

        *shmaddr = 'a';

        //char *args[] = {"./slave", NULL};
        char *args[] = {(char*)"python3", (char*)"vit_pipeline.py", argv[1], NULL};
        execvp(args[0], args); 
        

        

        // exit the second process
        exit(0);
    } else if (pid2 < 0) {
        perror("fork");
        exit(1);
    }

    // wait for both processes to finish
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    // detach and remove the shared memory segment
    shmdt(shmaddr);
    shmctl(shmid, IPC_RMID, NULL);




return 0;

}



