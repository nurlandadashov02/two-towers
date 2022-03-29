# authors: Nurlan Dadashov (2019400300), Aziza Mankenova (2018400387)
# Compiling
# Working
# Checkered-style

from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
P = comm.Get_size() - 1  # number of worker processes
x = np.sqrt(P)  # number of blocks in one side

if rank == 0:  # manager process
    # read the input file and obtain coordinates of all the "+"s and "o"s
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        N, W, T = list(map(int, lines[0].split(" ")))
        coords = [[list(map(int, c.split(" "))) for c in line.strip().split(", ")] for line in lines[1:]]

    size_of_block = int(N // x)  # size of block is (size_of_block x size_of_block)

    for i in range(P):
        comm.send((size_of_block, N, W), dest=i + 1, tag=10)

    grid = np.full(shape=(W, N, N), fill_value={"content": ".", "health": 0, "attack": 0})
    worker_data = np.empty(shape=(P, W, size_of_block, size_of_block), dtype=object)
    for w in range(W):  # filling the grid
        for i, coord_list in enumerate(coords[2 * w:2 * w + 2]):
            for coord in coord_list:
                grid[w][coord[0], coord[1]] = {"content": "o", "health": 6, "attack": 1} if i % 2 == 0 else {
                    "content": "+", "health": 8, "attack": 2}

        # splitting grid into (N/x, N/x) blocks
        blocks = np.array([np.hsplit(v, x) for v in np.vsplit(grid[w], x)])
        blocks = blocks.reshape(P, size_of_block, size_of_block)
        for p in range(P):
            worker_data[p, w] = blocks[p]

    # sending data to each worker process
    for i, data in enumerate(worker_data):
        comm.send(data, dest=i + 1, tag=10)

    all_blocks = np.empty((P, size_of_block, size_of_block), dtype=object)
    # worker processes receive data from manager process
    for p in range(P):
        bl = comm.recv(source=p + 1, tag=10)
        all_blocks[p] = bl

    # formatting all_blocks to print it easily
    all_blocks = np.array(np.vsplit(all_blocks, x))
    with open(sys.argv[2], "w+") as f:
        for row in all_blocks:
            for i in range(size_of_block):
                for c, col in enumerate(row):
                    for j in range(size_of_block):
                        f.write(col[i, j]["content"] + " ")
                f.write("\n")

else:  # [rank]th worker process code
    # size of block is (size_of_block x size_of_block)
    size_of_block, N, W = comm.recv(source=0, tag=10)
    data = comm.recv(source=0, tag=10)  # data.shape = (W, size_of_block, size_of_block)

    # position of the neighbor cell in the given direction
    position = {
        "top": int(rank - x),
        "bottom": int(rank + x),
        "left": int(rank - 1),
        "right": int(rank + 1),
        "top_left": int(rank - x - 1),
        "top_right": int(rank - x + 1),
        "bottom_left": int(rank + x - 1),
        "bottom_right": int(rank + x + 1)
    }
    # reverse position of each type of position element
    position_reverse = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
        "top_left": "bottom_right",
        "top_right": "bottom_left",
        "bottom_left": "top_right",
        "bottom_right": "top_left"
    }

    # getting list of neighbours given number of the cell and number of blocks in one side of the map
    def get_neighbors(_rank, _x):
        if _rank == 1:  # top-left
            _neighbors = ["right", "bottom", "bottom_right"]
        elif _rank == _x:  # top-right
            _neighbors = ["left", "bottom", "bottom_left"]
        elif _rank == _x ** 2 - _x + 1:  # bottom-left
            _neighbors = ["right", "top", "top_right"]
        elif _rank == _x ** 2:  # bottom-right
            _neighbors = ["left", "top", "top_left"]
        elif (_rank - 1) % _x == 0:  # left edges
            _neighbors = ["right", "top", "bottom", "top_right", "bottom_right"]
        elif _rank % _x == 0:  # right edges
            _neighbors = ["left", "top", "bottom", "top_left", "bottom_left"]
        elif _rank // _x == 0 and (_rank % _x != 0 or _rank % _x != 1):  # top edges
            _neighbors = ["left", "right", "bottom", "bottom_left", "bottom_right"]
        elif _rank // _x == _x - 1 and (_rank % _x != 0 or _rank % _x != 1):  # bottom edges
            _neighbors = ["left", "right", "top", "top_left", "top_right"]
        else:  # center
            _neighbors = ["left", "right", "top", "bottom", "top_left", "bottom_left", "top_right", "bottom_right"]
        return _neighbors

    neighbors = get_neighbors(rank, x)

    # data exchange with other workers
    def communicate(w):
        # which data should be sent to the neighbor in the given direction
        send_data_to_neighbor = {
            "top": data[w, 0, :],
            "bottom": data[w, -1, :],
            "left": data[w, :, 0],
            "right": data[w, :, -1],
            "top_left": data[w, 0, 0],
            "top_right": data[w, 0, -1],
            "bottom_left": data[w, -1, 0],
            "bottom_right": data[w, -1, -1]
        }

        neighbor_data = {}

        for neighbor in neighbors:
            if (rank % x != 0 and (rank // x) % 2 == 0) or (rank % x == 0 and (rank // x) % 2 == 1):
                if neighbor in ["bottom", "bottom_left", "bottom_right"]:
                    # even rows send data to bottom odd rows
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)
                    # even rows receive data from bottom odd rows
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
            else:
                if neighbor in ["top", "top_left", "top_right"]:
                    # odd rows receive data from even row above
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
                    # odd rows send data to even row above
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)

            if (rank % x) % 2 == 1:
                if neighbor in ["right"]:
                    # even columns send data to right odd columns
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)
                    # even rows receive data from right odd rows
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)

            else:
                if neighbor in ["left"]:
                    # odd columns receive data from left even columns
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
                    # odd columns send data to left even columns
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)

        for neighbor in neighbors:
            if (rank % x != 0 and (rank // x) % 2 == 0) or (rank % x == 0 and (rank // x) % 2 == 1):
                if neighbor in ["top", "top_left", "top_right"] and rank > x:
                    # even rows send data to top odd rows except the first row
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)
                    # even rows receive data from top odd rows except the first row
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
            else:
                if neighbor in ["bottom", "bottom_left", "bottom_right"] and rank <= x ** 2 - x:
                    # odd rows receive data from bottom even rows except the last row
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
                    # odd rows send data to bottom even rows except the last row
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)

            if (rank % x) % 2 == 1:
                if neighbor in ["left"] and rank % x != 1:
                    # even columns send data to left odd columns except the first column
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)
                    # even columns receive data from left odd columns except the first column
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)

            else:
                if neighbor in ["right"] and rank % x != 0:
                    # odd columns receive data from right even columns except the last column
                    another_row = comm.recv(source=position[neighbor], tag=10)
                    neighbor_data.update(another_row)
                    # odd columns send data to right even columns except the last column
                    comm.send({position_reverse[neighbor]: send_data_to_neighbor[neighbor]}, dest=position[neighbor], tag=10)

        return neighbor_data

    # finding opponent in neighbor cell (given direction)
    def get_opponent(data, neigb, i, j):
        if neigb == "top":
            if i == 0:
                op = neighbor_data[neigb][j]
            else:
                op = data[w, i - 1, j]
        elif neigb == "bottom":
            if i == size_of_block - 1:
                op = neighbor_data[neigb][j]
            else:
                op = data[w, i + 1, j]
        elif neigb == "left":
            if j == 0:
                op = neighbor_data[neigb][i]
            else:
                op = data[w, i, j - 1]
        elif neigb == "right":
            if j == size_of_block - 1:
                op = neighbor_data[neigb][i]
            else:
                op = data[w, i, j + 1]
        elif neigb == "top_left":
            if i == 0 and j == 0:
                op = neighbor_data["top_left"]
            elif i == 0:
                op = neighbor_data["top"][j - 1]
            elif j == 0:
                op = neighbor_data["left"][i - 1]
            else:
                op = data[w, i - 1, j - 1]
        elif neigb == "top_right":
            if i == 0 and j == size_of_block - 1:
                op = neighbor_data["top_right"]
            elif i == 0:
                op = neighbor_data["top"][j + 1]
            elif j == size_of_block - 1:
                op = neighbor_data["right"][i - 1]
            else:
                op = data[w, i - 1, j + 1]
        elif neigb == "bottom_left":
            if i == size_of_block - 1 and j == 0:
                op = neighbor_data["bottom_left"]
            elif i == size_of_block - 1:
                op = neighbor_data["bottom"][j - 1]
            elif j == 0:
                op = neighbor_data["left"][i + 1]
            else:
                op = data[w, i + 1, j - 1]
        elif neigb == "bottom_right":
            if i == size_of_block - 1 and j == size_of_block - 1:
                op = neighbor_data["bottom_right"]
            elif i == size_of_block - 1:
                op = neighbor_data["bottom"][j + 1]
            elif j == size_of_block - 1:
                op = neighbor_data["right"][i + 1]
            else:
                op = data[w, i + 1, j + 1]
        else:
            op = {"content": ".", "health": 0, "attack": 0}

        return op


    for w in range(W):
        for r in range(8):
            neighbor_data = communicate(w)  # exchanging data with other workers
            data_copy = np.copy(data)
            for i, row in enumerate(data[w]):
                for j, cell in enumerate(row):  # calculate damage to the cell from its neighbors
                    if cell["content"] != ".":
                        # calculating id of a cell by determining how many cells are above, left, right of it.
                        # 1 <= id <= N*N
                        A_of_blocks_above = (rank // x if rank % x != 0 else (
                                    rank // x - 1)) * size_of_block * size_of_block * x
                        A_of_blocks_left = ((rank % x) - 1 if rank % x != 0 else x - 1) * size_of_block * (i + 1)
                        A_of_blocks_itself = i * size_of_block + j
                        A_of_blocks_right = (x - (rank % x) if rank % x != 0 else 0) * size_of_block * i

                        id_of_cell = A_of_blocks_above + A_of_blocks_left + A_of_blocks_itself + A_of_blocks_right + 1
                        cell_neighbors = get_neighbors(id_of_cell, N)
                        for neigb in cell_neighbors:
                            op = get_opponent(data_copy, neigb, i, j)
                            if cell["content"] == "o":
                                if op["content"] == "+" and neigb in ["top", "bottom", "left", "right"]:
                                    data[w, i, j]["health"] -= op["attack"]

                            elif cell["content"] == "+":
                                if op["content"] == "o":
                                    data[w, i, j]["health"] -= op["attack"]

                            # remove tower if health = 0
                            if cell["content"] != "." and data[w, i, j]["health"] == 0:
                                data[w, i, j] = {"content": ".", "health": 0, "attack": 0}

        # placing new towers
        if w < W - 1:
            placement = np.copy(data[w + 1])
            data[w + 1] = data[w]
            for i, row in enumerate(data[w + 1]):
                for j, cell in enumerate(row):
                    if cell["content"] == "." and placement[i, j]["content"] != ".":
                        data[w + 1, i, j] = placement[i, j]

    # sending result of last wave to manager
    comm.send(data[-1], dest=0, tag=10)
