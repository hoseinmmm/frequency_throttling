import shutil
import os
import time
import subprocess
import psutil
import datetime
import statistics


def set_cpu_affinity(core_number):
    pid = os.getpid()  # Get the process ID of the current Python program
    p = psutil.Process(pid)
    p.cpu_affinity([core_number])  # Set the CPU affinity to the specified core


def cal_mean(f_name):
    for f_n in f_name:
        f = open(f_n, "r")
        text1 = f.read()
        f.close()
        mylist = list(text1.split("\n"))

        my_int_list = [int(a,0) for a in mylist if a!='']

        std = statistics.pstdev(my_int_list[1000:])
        mean = statistics.mean(my_int_list[1000:])
        may_max = max(my_int_list)

        print("file: " + f_n + ", MEAN: " + str(mean) + ", STD: " + str(std)  + ", max: " + str(may_max))

        return mean, std



def create_directory(directory_name):
    current_path = os.getcwd()
    new_directory_path = os.path.join(current_path, directory_name)

    if os.path.exists(new_directory_path):
        # Remove the existing directory and its contents
        #shutil.rmtree(new_directory_path)
        print(f"Existing directory '{directory_name}'.")
    else:
        # Create the new directory
        #os.mkdir(new_directory_path)
        os.makedirs(new_directory_path)
        print(f"Directory '{directory_name}' created.")


def move_text_file(source_file, destination_path):
    
    create_directory(destination_path)
    try:
        shutil.move(source_file, destination_path)
        print(f"File '{source_file}' moved to '{destination_path}' successfully.")
    except FileNotFoundError:
        print(f"Source file '{source_file}' not found.")
    except PermissionError:
        print(f"Permission denied to move file '{source_file}' to '{destination_path}'.")
    except Exception as e:
        print(f"An error occurred while moving the file: {e}")



def settle_sys():
    limit= 0.7#5
    while True:
        load1, load5, load15 = os.getloadavg()
        print(load1,load5,load15)
        if load1 <= limit:
            break
        print('Please run me only on an idle system.')
        print('Found 1 minute load average of %.2f' % load1)
        print('Expecting max 1 minute load average of %.2f' % limit)
        print('This is protection against old measurement processes\nlingering and corrupting new measurements.')
        print('Please check for lingering measurement processes. Trying again.')
        print('')
        time.sleep(5)


def acquire(model_sel='1',inst='1',result_path='results_test/test1',rounds='0'):

    current_dir = os.getcwd()
    exe = current_dir + '/orchestrator_vit'
    file_name=f'raw_result_model_{model_sel}_inst_{inst}_r_{rounds}.txt'
    
    invoke_args = [ exe, model_sel, file_name, inst]
    print(invoke_args)

    print("Calling subprocess to run c codes:")
    #subprocess.call(invoke_args)

    #---------
    success = 0
    retry = 5
    import signal

    while (success==0) and (retry>0):
        # Start the subprocess using subprocess.Popen
        settle_sys()
        process = subprocess.Popen(invoke_args)
        # Wait for a specific amount of time
        timeout_seconds = 10 * 60  # Adjust this according to your needs
        try:
            process.wait(timeout=timeout_seconds)
            success = 1
            print("Success :)")
        except subprocess.TimeoutExpired:
            print("Failed to finish ...")
            retry = retry - 1
            # If the subprocess is still running after the timeout, terminate it
            process.terminate()
            # Optionally, you can also try process.kill() if terminate() doesn't work

            # Wait for a few seconds to ensure the subprocess is terminated
            process.wait(timeout=5)

            # Check if the process is still alive
            if process.poll() is None:
                # If the process is still alive, use os.kill to force termination
                process.kill()
    #---------

    if success==1:
        mean,std = cal_mean([file_name])

        move_text_file(current_dir+'/'+file_name, result_path)
    else:
        mean=0
        std=0

    return mean,std


if __name__ == "__main__":

    settle_sys()

    desired_core = 4  # Set the desired CPU core number
    set_cpu_affinity(desired_core)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")

    models = [1,2,3,5,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36]
    instructions = [1,2,3,4,5,6]
    rounds = 200
    cores = '02'
    power_limit = '122'

    features = ""
    for i in range(1, rounds+1):
        features += f"mean{i},std{i},"
    features = features.rstrip(',') 

    dataset = 'Model,Instruction,' + features + '\n'

    line = 'Model,Instruction,Mean,STD\n'

    zeros = 0
    for m in models:
        for i in instructions:
            mean_l = []
            std_l = []
            if (m in []) and (i in []):
                continue

            dataset = dataset + str(m) + ',' + str(i) + ','
            for r in range(rounds):
                
                mean,std = acquire(str(m),str(i),f'regression_results/t_{time_str}/c{cores}_p{power_limit}_model_{m}',str(r))
                mean_l.append(mean)
                std_l.append(std)
                if mean==0:
                    zeros = zeros + 1
                dataset = dataset + str(mean) + ',' + str(std) + ','
                 
            dataset = dataset + '\n'
            mean = statistics.mean(mean_l)
            std = statistics.mean(std_l)
            line = line + str(m) + ',' + str(i) + ',' + str(mean) + ',' + str(std) + '\n'
            print(line)

            this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            f = open(this_dir + f'/regression_results/t_{time_str}/summary.csv', "w")
            f.write(line)
            f.close()

            f = open(this_dir + f'/regression_results/t_{time_str}/dataset.csv', "w")
            f.write(dataset)
            f.close()
    print("Number of failed elements: ", zeros)