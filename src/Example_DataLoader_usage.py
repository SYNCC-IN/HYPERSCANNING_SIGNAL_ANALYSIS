from src.DataLoader import DataLoader
# creates a DataLoader class object that creates a structure described in the docs folder (data_structure_spec.md) from raw data
data = DataLoader("W_010","../DATA/W_010",None,None,output_dir="../DATA/W_010/",plot_flag=True)
# usage of staticmethod load_output_data loads data created by DataLoader
DataLoader.load_output_data("../DATA/W_010/W_010.joblib")