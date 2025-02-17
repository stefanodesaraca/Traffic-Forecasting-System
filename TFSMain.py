

































def main():
    while True:
        print("""==================== MENU ==================== 
                 1. Set pre-analysis information
                    1.1 Create an operation
                    1.2 Set an operation as active (current one)
                 2. Download traffic volumes data (Trafikkdata API)
                 3. Forecast
                    3.1 Set forecasting target datetime
                    3.2 Forecast warmup
                    3.3 Execute forecast
                        3.3.1 One-Point Forecast
                        3.3.2 A2B Forecast
                 4. Road network graph generation
                 5. Other options
                    5.1 EDA (Exploratory Data Analysis)
                    5.2 Erase all data about an operation
                 0. Exit""")

        option = input("Choice: ")
        print()

        if option == "1":
            pass

        elif option == "0":
            exit()

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























