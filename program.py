from sudoku import load_config
from sudoku.gui import GUI

def main():
    config_path = "./configs/program.yml"
    config = load_config(config_path)
    gui = GUI(config)
    gui.run()

main()
