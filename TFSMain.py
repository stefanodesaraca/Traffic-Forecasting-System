from OpsSettings import *


def manage_ops(functionality: str):

    if functionality == "1.1":
        ops_name = input("Insert new operation name: ")
        create_ops_folder(ops_name)

    if functionality == "1.2":
        ops_name = input("Insert the operation to set as active: ")
        write_active_ops_file(ops_name)

    if functionality == "1.3":
        print("Active operation: ", read_active_ops_file(), "\n\n")

    else:
        print("Functionality not found, try again with a correct one")
        print("Returning to the main menu...")
        main()
        
    return None





























def main():
    while True:
        print("""==================== MENU ==================== 
                 1. Set pre-analysis information
                    1.1 Create an operation
                    1.2 Set an operation as active (current one)
                    1.3 Check the active operation name
                 2. Download traffic volumes data (Trafikkdata API)
                 3. Forecast
                    3.1 Set forecasting target datetime
                    3.2 Forecast warmup
                    3.3 Execute forecast
                        3.3.1 One-Point Forecast
                        3.3.2 A2B Forecast
                 4. Road network graph generation
                 5. Other options
                    5.1 Set forecasting system folders (manually)
                    5.2 EDA (Exploratory Data Analysis)
                    5.3 Erase all data about an operation
                    5.4 Find best model for the current operation
                    5.5 Analyze pre-existing road network graph
                 0. Exit""")

        option = input("Choice: ")
        print()

        if option == "1.1":
            manage_ops("1.1")

        if option == "1.2":
            manage_ops("1.2")

        if option == "1.3":
            manage_ops("1.3")

        elif option == "0":
            exit()

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























