def counting_islands(matrix_, m, n):
    islands = 0
    for i in range(m):
        for j in range(n):
            if not visited[i][j] and matrix_[i][j] == 1:
                dfs(matrix_, i, j)
                islands += 1  # each time we start a new dfs we start exploring new island
    return islands


def dfs(matrix_, i, j):
    if i < 0 or j < 0 or i >= len(matrix_) or j >= len(matrix_[0]) or matrix_[i][j] != 1:
        return
    if not visited[i][j]:
        visited[i][j] = True
        dfs(matrix_, i + 1, j)  # up
        dfs(matrix_, i - 1, j)  # down
        dfs(matrix_, i, j + 1)  # right
        dfs(matrix_, i, j - 1)  # left


rows, columns = map(int, input().split())
visited = [[False for _ in range(columns)] for _ in range(rows)]
matrix = []
for _ in range(rows):
    matrix.append([int(x) for x in input().split()])

print(counting_islands(matrix, rows, columns))
