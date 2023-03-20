class SolveSudoku:
    
    def __init__(self, grid) -> None:
        self.grid = grid
    
    def valid(self, grid, num, pos) -> bool:
        # Check row
        for i in range(len(grid[0])):
            if grid[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(grid)):
            if grid[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if grid[i][j] == num and (i, j) != pos:
                    return False

        return True 
    
    def find_empty(self, grid) -> None:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return (i, j)  # row, col

        return None
    
    def solve(self) -> bool:
        find = self.find_empty(self.grid)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if self.valid(self.grid, i, (row, col)):
                self.grid[row][col] = i

                if self.solve():
                    return True

                self.grid[row][col] = 0

        return False    