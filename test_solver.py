from puzzle_solver import PuzzleSolver
def main():
    puzzle = []
    with open('tree_search.txt', 'r') as f:
        line = f.readline().lower()
        while line:
            puzzle.append(line.split(','))
            puzzle[-1][-1] = puzzle[-1][-1][:-1]
            line = f.readline().lower()
    
    bank = []
    with open('tree_bank.txt', 'r') as f:
        line = f.readline()
        while line:
            bank.append(line[:-1].lower())
            line = f.readline()

    solver = PuzzleSolver(len(puzzle), len(puzzle[0]), puzzle, bank)
    solver.solve()

if __name__ == '__main__':
    main()
