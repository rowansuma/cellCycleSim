import subprocess
import sys

from FISS.run import Trial

def main():
    plot_proc = subprocess.Popen([sys.executable, "FISS/plot.py"])

    num_trials = 1
    for i in range(num_trials):
        trial = Trial(
            display_visuals=True,
            display_cells=True,
            display_phase=True,
            display_ecm=True,
            scalpel="circle"
        )
        trial.run_trial()

    plot_proc.terminate()


if __name__ == '__main__':
    main()