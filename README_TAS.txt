1. Create a folder for the project and make two folders in there (one named "data_files", and the second "main")

2. Unzip the data_files folder and copy the dewarping code and the NACA file in the data_files folder

3. Unzip the tas_repo folder and copy all codes in the "main" folder

4. Put your TAS_DATA.h5 file in the data_files folder (the big 4gb one)

5. Check all scripts for any modules you dont have installed yet and pip install them.

6. Run the data_dewarping.py code. This will take about 4-5 minutes and will write a new file "dewarped_data.npz" in the data_files folder.

7. Now you should be able to run any other code in the "main" folder and instead of sending the plots to discord you can just plt.show() on your local device.
