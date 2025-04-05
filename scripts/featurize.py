#NOTE: This script will only run successfully after you install Ersilia Model Hub and fetch the model of interest

from ersilia import ErsiliaModel
import os
import pandas as pd


def featurize(model_ID, dataset_path, smiles_column):
    mdl = ErsiliaModel(model_ID)
    mdl.serve()

    #load data
    df = pd.read_csv(f"{dataset_path}") 

    # Extract just the Drug column containing SMILES
    smiles_df = df[[smiles_column]]

    # Save smiles to new CSV file
    smiles_df.to_csv("smiles_only.csv", index=False) 

    #create an empty CSV file to store featurized data
    pd.DataFrame().to_csv('featurized_data.csv', index=False)

    datasets = {
            "smiles_only.csv": "featurized_data.csv",
        }
    
    for input_file, output_file in datasets.items():
        # Check if the input file exists
        if os.path.exists(input_file): 

            # Run the model/featurization on the input file
            # Generating output to the specified output file
            mdl.run(input=input_file, output=output_file)
        else:
            # Raises an error if the input file is missing
            raise FileNotFoundError(f"Input file '{input_file}' not found!")  

    try:
        os.remove("smiles_only.csv")
    except FileNotFoundError:
        print(f"File not found")
        mdl.close()
    except PermissionError:
        print(f"Permission denied to delete file")
        mdl.close()
    except Exception as e:
        print(f"Error deleting file: {e}")
        mdl.close()

    #close served model
    mdl.close() 

    print("Featurization Complete!")


featurize("eos5axz", "../data/external_data.csv", "Smile")
