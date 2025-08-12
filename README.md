# Cell Cycle Simulation
Made by Rowan Sumanaweera

## Goal

<p>To make a rudimentary simulation of the cell cycle that can later be used to model gene expression.</p>

## Necessary Components/Rules (for now)

* Cells must undergo a cell cycle (proportional to real world timings)
* Cells must divide at the end of said cycle to reproduce
* Cells must undergo contact inhibition and exit the cell cycle when in contact with many other cells
* Cells must have physics interactions to prevent them from overlapping

## Current Task List

- [x] Research & Implement Verlet Integration
- [x] Draw Cells
- [x] Optimize with Spatial Partitioning
- [x] Allow for Appending and Removing Cells from Field
- [x] Add Cell Cycle
- [x] Add Contact Inhibition
- [x] Fix Border Cell Division
- [x] Add Graphing Tool
- [x] Add Deletion Tool
- [x] Implement Cell Cycle Stages
- [x] Tweak Contact Inhibition
- [x] Add Fibroblast Motility
- [x] Add Gene Expression
- [x] Add ECM Particles and ECM Behavior
- [x] Add Variable Cell Motion
- [x] Add Input Parameters TOML File
- [ ] Add Deletion Shapes
- [ ] Add Output Diagnostic Page & Data Storage
- [ ] Add ECM Plot, Cell Inhibition Time Plot
- [ ] Make Cells Elliptical
- [ ] Make Gene Expression Peaks Variable
- [ ] Modify Gene Expression to Use Real Data
- [ ] Make Gene Expression Determine Cell Cycle

## Usage

1. Run plot.py and plot2.py to create MatPlotLib window for real-time graphing
2. Run main.py to open the simulation, input simulation parameters.
3. Hold down Left Click to delete cells within a radius
4. Press Right Click to create a cell
5. Note that the colors indicate cell cycle stage; (gray=G0, blue=G1, yellow=S, green=G2, red=M)

# Dependencies
The simulation was designed using python 3.9 and the following python packages: taichi 1.7.3, tomli 2.2.1, numpy 2.0.2, matplotlib 3.9.4, seaborn 0.13.2, and pandas 2.3.0.

## Other Notes

* I originally was planning to use standard python classes and Verlet integration to do this, but after some research I decided to try using [Taichi](https://www.taichi-lang.org/). This will allow me to use Verlet integration with help from the GPU and speed things up.