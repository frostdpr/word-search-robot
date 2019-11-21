from collections import defaultdict
import sys

class PuzzleSolver:

    def __init__(self, rows, cols, puzzle, bank):
        self.r= rows
        self.c= cols
        self.puzzle = puzzle
        self.min_word_len = 23
        self.max_word_len = 0
        self.load_word_bank(bank)
    
    #Takes in a list of strings to insert into wordbank dictionary
    def load_word_bank(self, bank):
        self.bank = defaultdict(lambda: None)

        for word in bank:
            self.min_word_len = min(self.min_word_len, len(word))
            self.max_word_len = max(self.max_word_len, len(word))
            self.bank[word] = word

    #switch case for direction
    def get_word_in_grid(self, startRow, startCol, d, l):
        word = ""
        pos, r, c = 0, startRow, startCol

        for i in range(l):
            #if current row or column is out of bounds, break
            if c >= self.c or r >= self.r or r < 0 or c < 0:
                return ""

            #pos += 1
            #print(self.puzzle[r][c])
            word += self.puzzle[r][c]

            if d == 0:
                r -= 1
            elif d == 1:
                r -=1
                c += 1
            elif d == 2:
                c += 1
            elif d == 3:
                r += 1
                c += 1
            elif d == 4:
                r += 1
            elif d == 5:
                r += 1
                c -= 1
            elif d == 6:
                c -= 1
            elif d == 7:
                r -= 1
                c -= 1
            
        return word

    def solve(self):
        word_count = 0
        not_found = list(self.bank.keys())
        print('\n\n')
        print('puzzle bank', not_found)
        for i in range(self.r):
            for j in range(self.c):
                for d in range(8):
                    for length in range(self.min_word_len, self.max_word_len+1):
                        word = self.get_word_in_grid(i, j, d, length)

                        if self.bank[word] == word:
                            print(i, j, word)
                            try:
                                not_found.remove(word)
                            except:
                                pass

                            word_count += 1

        print('words found', word_count)
        return not_found
