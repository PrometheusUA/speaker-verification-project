# Speaker verification project

**by Andrii Shevtsov, Artur Shevtsov and Maksym Palamariuk**

This is a realization of speaker verification system using pure numpy (PLDA technique) and its comparison with more advanced methods, such as Siamese Convolutional Neural Networks.

## Folder structure

Folder structure is as follows:
- `data` folder is folder to store the dataset.
- `features` folder is for feature extraction. Those features are mainly used in PLDA realization.
- `neural` folder is for neural network-based Speaker verification system realization.
- `plda` folder is for realization of the PLDA algorithm and its enhancements.
- `utils` folder is for other project-related stuff that is general or inconvinient to put into previous folders.

## Environment

We are using Python's narive virtual environments along with pip package manager and `requirements.txt` files.

To run the project, create a virtual environment via `python<version> -m venv .venv`.

Then, activate the environment:
- On Linux/Mac: `source .venv/bin/activate`.
- On Windows: `.venv/Scripts/activate.bat`.

And install `requirements.txt`:
```
pip install -r requirements.txt
```

If you want to use neural net, you need also to install `neural/requirements.txt`.

## Dataset

To obtain the dataset, use the following steps:
1. Create a `data` folder in the project.
<!-- 3. Ask for the dataset access via https://cn01.mmai.io/keyreq/voxceleb. We will only need VoxCeleb1 dataset, as it is smaller.
<!-- 3. Run `./utils/dataprep.py` file (taken from [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) repo, with minor modifications) like this: `python ./utils/dataprep.py --save_path data --user <USERNAME> --password <PASSWORD>`. **Note:** Don't forget to create an environment and install the dependencies!
4. Download all parts and concatenate dev files using the command `cat vox1_dev* > vox1_dev_wav.zip`.
5. Unzip all the files into the `data` folder. -->
2. Run all cells in `./utils/download_data.ipynb`
