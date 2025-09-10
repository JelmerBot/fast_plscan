# Elegans dataset

This dataset was created by [Packer et
al.](https://www.science.org/doi/10.1126/science.aax1971), showing how
single-cell RNA sequencing uncovers trajectories relating to the cell
development in the *C. elegans*.

Run these scripts to reproduce the data files used in the documentation
notebooks:

- `elegans.r`: Downloads and pre-processes the data. It follows the instructions
at https://cole-trapnell-lab.github.io/monocle3/docs/trajectories/.
 
- `to_Xy.py`: Converts the data into a feature matrix `generated/X_elegans.npy`
  and target vector `generated/y_elegans.npy`.