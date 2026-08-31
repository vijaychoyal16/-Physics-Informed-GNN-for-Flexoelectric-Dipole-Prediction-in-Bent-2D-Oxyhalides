## Workflow

1. Generate bent structures with different bending amplitudes.
2. Run VASP Berry-phase calculations to obtain dipole/polarization labels.
3. Extract labels and physical descriptors:
   - Px, Py, Pz
   - bending amplitude
   - strain gradient
   - force maximum and mean force
   - stress tensor components
   - optional dielectric tensor and Born effective charges
4. Convert atomic structures into graph representations.
5. Train a GNN to predict dipole/polarization response.
6. Visualize atomic dipole vectors using quiver plots.
7. Calculate flexoelectric coefficients and compare DFT vs GNN trends.

## Main Scripts

- `01_prepare_dataset_extended.py`  
  Prepares the dataset CSV by linking structure files with target labels and physical descriptors.

- `02_build_graph_extended.py`  
  Converts VASP structures into PyTorch Geometric graph files.

- `03_model_extended.py`  
  Defines the GNN model using atomic graph information and global physical features.

- `04_train.py`  
  Trains the GNN model.

- `05_evaluate.py`  
  Evaluates the trained model.

- `06_predict.py`  
  Predicts total and atomic dipoles for new structures.

- `07_visualize_atomic_dipoles.py`  
  Visualizes predicted atomic dipoles using 2D/3D quiver plots.

- `09_flexoelectric_coefficient.py`  
  Calculates flexoelectric coefficients from polarization and strain-gradient data.

- `12_plot_strain_gradient_vs_dipole.py`  
  Plots DFT vs GNN dipole trends as a function of strain gradient.

- `14_calculate_young_modulus.py`  
  Estimates effective Young's modulus from stress-strain or bending-derived strain.

## Example Dataset Format

```csv
id,Px,Py,Pz,amplitude,strain_gradient,force_max,force_mean,stress_xx,stress_yy,stress_zz,stress_xy,stress_yz,stress_zx
FeOBr_0.1,0.0,0.0,0.00000000,0.1,2.0e5,0.1,0.03,-1,2,3,0,0,0
FeOBr_0.2,0.0,0.0,0.00010000,0.2,4.0e5,0.1,0.04,-1,2,3,0,0,0

python 01_prepare_dataset_extended.py --label_csv id_prop_extended_clean.csv
python 02_build_graph_extended.py --cutoff 5.0 --inspect
python 03_model_extended.py --graph_path processed/graphs/FeOBr_0.1.pt --device cpu
python 04_train.py --device cpu --epochs 3000 --hidden_dim 64 --num_layers 3 --lr 5e-4
python 05_evaluate.py --device cpu
