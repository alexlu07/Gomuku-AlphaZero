import numpy as np

class Env:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.board_area = self.board_size ** 2

        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = None

        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]
        
        self.available = []
        self.last_move = [None, None]

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        self.available = list(range(self.board_area))

    def step(self, action):
        assert action in self.available
        self.available.remove(action)

        x = action // self.board_size
        y = action % self.board_size
        self.board[x][y] = self.player
        self.last_move = [x, y]

        self.player *= -1

    def get_observation(self):
        obs = np.zeros((4, self.board_size, self.board_size), dtype="float32")
        obs[0][self.board == self.player] = 1
        obs[1][self.board == -self.player] = 1
        obs[2][self.last_move[0]][self.last_move[1]] = 1
        obs[3] = self.player
        return obs

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j

                    x_end, y_end = x + 4*d[0], y + 4*d[0]
                    if (x_end < 0 or x_end >= self.board_size or 
                        y_end < 0 or y_end >= self.board_size): break

                    count = 0
                    for _ in range(5):
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True, player
        return not has_legal_actions, 0

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = action // self.board_size
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y