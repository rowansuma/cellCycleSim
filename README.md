<p align="center">
  <img src="./figures/fiss-logo.png" width="250">
</p>
<h1 align="center">FISS: A Fibroblast In-Silico Simulator</h1>
<p align="center" style="margin-top: -10px;">
  <em></em>
</p>
<p align="center">
<a href="https://docs.google.com/document/d/1ROzoR6iv61_18U8YB8cceNsBOWdE905ZzzW_XGZZEgM/edit?tab=t.0"><img src="https://img.shields.io/badge/Summary-white?style=for-the-badge&logo=googledocs" alt="Summary"></a>
<a href="https://docs.google.com/presentation/d/1ez3Qw04UkxXO1HXnFAF8e8aQD9lcpfGfwMf-Ycy0IGI/edit?slide=id.p#slide=id.p"><img src="https://img.shields.io/badge/Slideshow-white?style=for-the-badge&logo=googleslides" alt="Slideshow"></a>
<a href="https://drive.google.com/file/d/1KrZHcmWfxkzDpafXBIXLX_A5S66oK6V5/view?t=5"><img src="https://img.shields.io/badge/Demonstration-white?style=for-the-badge&logo=youtube&logoColor=ff0000" alt="Demonstration"></a>
</p>

<p>A Fibroblast migration and proliferation simulation to model Wound Healing, Gene Expression, and the Cell Cycle.
  
  Made by Rowan Sumanaweera. Project started in June 2025.</p>

## Key Components

Fibroblasts:
* are represented by large colored circles 
* move along a random movement vector when not in G0
* go through the cell cycle (gray = G0, blue = G1, yellow = S, green = G2, red = M), and divide after M
* can exit the cell cycle and enter G0 if too crowded (Contact Inhibition) and can re-enter if the nearby cell density decreases
* occasionally deposit ECM (extra-cellular matrix) "particles" (represented by small purple circles)
* are repelled by ECM to encourage them to close wounds

## Related Figures
<table align="center">
  <tr>
    <td>
      <img src="./figures/cell-growth.gif" height="180">
    </td>
    <td>
      <img src="./figures/circle-wound-healing.gif" height="180">
    </td>
    <td>
      <img src="./figures/triangle-wound-healing.gif" height="180">
    </td>
    <td>
      <img src="./figures/square-wound-healing.gif" height="180">
    </td>
    <td>
      <img src="./figures/line-wound-healing.gif" height="180">
    </td>
  </tr>
</table>
<p><em>Various wound shape presets for scratch assay experiments.</em></p>

<img src="./figures/ecm-fibroblasts.png" height="220">
<p><em>ECM and Fibroblast growth at a given simulation step.</em></p>

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
- [x] Add Deletion Shapes
- [x] Create Image Frame Capturer
- [x] Design Experimental Data Collection Methods
- [x] Begin Validation
- [ ] Write Paper
- [ ] Publish

## Tools
* **Space**: pause simulation
* **Right** click: create cell
* **Shift** + left click: delete cells within a specialized radius and shape (only while paused)
* **Alt**: save simulation state
* **Escape**: exit

<p>Input your specialized experiment configuration in config.toml.
<p>Find your saved simulation state in /savestates/

## Dependencies
The simulation was designed using python 3.9 and the following python packages: taichi 1.7.3, tomli 2.2.1, numpy 2.0.2, matplotlib 3.9.4, seaborn 0.13.2, and pandas 2.3.0.
