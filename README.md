# Cell Cycle Simulation
Made by Rowan Sumanaweera

## Goal

<p>To make a rudimentary simulation of the cell cycle that can later be used to model gene expression.</p>

## Necessary Components/Rules (for now)

* Cells must undergo a cell cycle (proportional to real world timings)
* Cells must divide at the end of said cycle to reproduce
* Cells must undergo contact inhibition and exit the cell cycle when in contact with many other cells
* Cells must have physics interactions to prevent them from overlapping

## The Approach

<p>I want to handle this project in multiple stages.</p>

1. Demonstrate all of the above rules while representing cells as simple rigid circles
2. Move on to rigid ovals (possibly could skip this step)
3. Move on to flexible cell membranes
   
<p>Once I get to the third stage, I can start worrying about gene expression and the future plans for this project.</p>

## Current Task List

- [x] Research & Implement Verlet Integration
- [x] Draw Cells
- [x] Optimize with Spatial Partitioning
- [x] Allow for Appending and Removing Cells from Field
- [x] Add Cell Cycle
- [x] Add Contact Inhibition
- [x] Fix Border Cell Division
- [ ] Add Deletion Tool
- [ ] Implement Cell Cycle Stages
- [ ] Implement True Contact Inhibition
- [ ] Move on from circles?

## Other Notes

* I originally was planning to use standard python classes and verlet integration to do this, but after some research I decided to try using [Taichi](https://www.taichi-lang.org/). This will allow me to use verlet integration with help from the GPU and speed things up.