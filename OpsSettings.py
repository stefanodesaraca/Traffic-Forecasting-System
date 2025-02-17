import os
from cleantext import clean

current_ops_filename = "current_ops"

def write_current_ops_file(ops_name: str):

    ops_name = clean(ops_name, to_ascii=True, no_emoji=True)

    with open(f"{current_ops_filename}.txt", "w") as ops_file:
        ops_file.write(ops_name)

    return None



#Reading operations file, it indicates which road network we're taking into consideration
def read_current_ops_file():

    try:
        with open(f"{current_ops_filename}.txt", "r") as ops_file:
            op = ops_file.read()

    except FileNotFoundError:
        print("\033[91mOperations File Not Found\033[0m")

    return op



def create_ops_folder(ops_name: str):

    ops_name = clean(ops_name, to_ascii=True, no_emoji=True)
    os.makedirs(ops_name, exist_ok=True)

    return None



def del_ops_folder(ops_name: str):

    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")

    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")


    return None

















































